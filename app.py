"""
🧠 Analizador IA de Noticias
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import gradio as gr
from newspaper import Article
from transformers import (
    pipeline,
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from langdetect import detect
import datetime
import pandas as pd
import torch
import re
import sqlite3
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import plotly.express as px

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
DB_PATH            = "news_cache.db"
LOG_PATH           = "news_analyzer.log"
CACHE_EXPIRY_DAYS  = 7          # días antes de re-analizar una URL
MAX_BATCH_WORKERS  = 2          # hilos en paralelo para lote

DOMINIOS_CONFIABLES = [
    "bbc.com", "reuters.com", "apnews.com",
    "elpais.com", "elmundo.es", "20minutos.es", "abc.es",
    "nytimes.com", "theguardian.com", "lemonde.fr",
]

TOPICS = [
    "política y gobierno",
    "economía y finanzas",
    "tecnología e innovación",
    "salud y medicina",
    "deportes",
]

BIAS_LABELS = ["neutral", "sensacionalista", "opinativo", "propagandístico"]

SENTIMIENTO_MAP = {
    "Muy negativo": 1, "Negativo": 2, "Neutral": 3,
    "Positivo": 4,     "Muy positivo": 5,
}

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logger.add(LOG_PATH, rotation="10 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# ─────────────────────────────────────────────
#  CARGA DE MODELOS
# ─────────────────────────────────────────────
logger.info("Cargando modelos de IA…")

summarizer           = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_tokenizer = summarizer.tokenizer

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
bias_classifier  = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ner_pipeline     = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

en_to_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_to_es_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
es_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
es_to_en_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

logger.info("Modelos cargados correctamente.")

# ─────────────────────────────────────────────
#  BASE DE DATOS SQLite
# ─────────────────────────────────────────────
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS noticias (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_registro   TEXT,
            url              TEXT UNIQUE,
            titulo           TEXT,
            fecha_articulo   TEXT,
            idioma           TEXT,
            tema             TEXT,
            resumen          TEXT,
            sentimiento      TEXT,
            sesgo            TEXT,
            credibilidad     REAL,
            entidades        TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def guardar_en_db(entry: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO noticias
            (fecha_registro, url, titulo, fecha_articulo, idioma,
             tema, resumen, sentimiento, sesgo, credibilidad, entidades)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        entry["fecha_registro"], entry["url"], entry["titulo"],
        entry["fecha_articulo"], entry["idioma"], entry["tema"],
        entry["resumen"], entry["sentimiento"], entry["sesgo"],
        entry["credibilidad"], json.dumps(entry["entidades"], ensure_ascii=False),
    ))
    conn.commit()
    conn.close()


def buscar_en_cache(url: str) -> dict | None:
    """Devuelve la fila si existe y no ha expirado."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.execute("SELECT * FROM noticias WHERE url = ?", (url,))
    row  = cur.fetchone()
    conn.close()
    if not row:
        return None

    cols = ["id","fecha_registro","url","titulo","fecha_articulo","idioma",
            "tema","resumen","sentimiento","sesgo","credibilidad","entidades"]
    fila = dict(zip(cols, row))

    fecha_reg = datetime.datetime.fromisoformat(fila["fecha_registro"])
    if (datetime.datetime.now() - fecha_reg).days > CACHE_EXPIRY_DAYS:
        return None   # expirado → re-analizar

    fila["entidades"] = json.loads(fila["entidades"])
    return fila


def obtener_historial() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT fecha_registro, titulo, tema, sentimiento,
                  sesgo, credibilidad, url
           FROM noticias
           ORDER BY fecha_registro DESC""",
        conn,
    )
    conn.close()
    return df


# ─────────────────────────────────────────────
#  FUNCIONES DE ANÁLISIS
# ─────────────────────────────────────────────
def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(
        r"(suscríbete|subscribe|cookies|privacy policy)",
        "", texto, flags=re.IGNORECASE,
    )
    return texto.strip()


def truncar_por_tokens(texto: str, tokenizer, max_tokens: int = 1024) -> str:
    tokens = tokenizer(texto, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)


def traducir(texto: str, origen: str, destino: str) -> str:
    if origen == destino:
        return texto
    if origen == "en" and destino == "es":
        tok, mdl = en_to_es_tokenizer, en_to_es_model
    elif origen == "es" and destino == "en":
        tok, mdl = es_to_en_tokenizer, es_to_en_model
    else:
        return texto
    inputs     = tok(texto, return_tensors="pt", truncation=True, padding=True)
    translated = mdl.generate(**inputs)
    return tok.decode(translated[0], skip_special_tokens=True)


def analizar_sentimiento(texto: str) -> str:
    inputs = sentiment_tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    estrellas = int(torch.argmax(outputs.logits)) + 1
    return ["Muy negativo","Negativo","Neutral","Positivo","Muy positivo"][estrellas - 1]


def clasificar_tema(texto: str) -> str:
    result = topic_classifier(texto, TOPICS)
    label  = result["labels"][0].lower()
    if "política"   in label: return "Política"
    if "economía"   in label: return "Economía"
    if "tecnología" in label: return "Tecnología"
    if "salud"      in label: return "Salud"
    if "deportes"   in label: return "Deportes"
    return "Otros"


def detectar_sesgo(texto: str) -> str:
    result = bias_classifier(texto[:512], BIAS_LABELS)
    return result["labels"][0].capitalize()


def extraer_entidades(texto: str) -> dict:
    resultados = ner_pipeline(texto[:512])
    entidades: dict[str, list] = {"PER": [], "ORG": [], "LOC": []}
    for ent in resultados:
        grupo   = ent.get("entity_group", "")
        palabra = ent.get("word", "").strip()
        if grupo in entidades and palabra and palabra not in entidades[grupo]:
            entidades[grupo].append(palabra)
    return entidades


def calcular_credibilidad(texto: str, fecha_articulo: str, url: str) -> float:
    score = 50.0

    # Longitud del artículo
    if len(texto) > 2000:
        score += 15
    elif len(texto) > 800:
        score += 8

    # Citas / fuentes mencionadas
    fuentes_kw = [
        "según", "de acuerdo con", "fuentes", "declaró", "afirmó",
        "according to", "sources", "said", "stated", "confirmed",
    ]
    menciones = sum(1 for kw in fuentes_kw if kw.lower() in texto.lower())
    score += min(menciones * 3, 15)

    # Fecha reciente
    if fecha_articulo != "Sin fecha":
        try:
            fecha     = datetime.datetime.strptime(fecha_articulo, "%Y-%m-%d")
            dias_diff = (datetime.datetime.now() - fecha).days
            if dias_diff < 7:    score += 10
            elif dias_diff < 30: score += 5
        except ValueError:
            pass

    # Dominio conocido
    dominio = urlparse(url).netloc.replace("www.", "")
    if any(d in dominio for d in DOMINIOS_CONFIABLES):
        score += 10

    return round(min(score, 100), 1)


# ─────────────────────────────────────────────
#  EXTRACCIÓN DE TEXTO CON FALLBACK
# ─────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _newspaper_download(url: str) -> Article:
    article = Article(url)
    article.download()
    article.parse()
    return article


def extraer_texto_articulo(url: str) -> tuple:
    """
    Devuelve (titulo, fecha, texto).
    Intenta newspaper3k con reintentos; si falla usa trafilatura.
    """
    titulo, fecha, texto = None, "Sin fecha", None

    # ── Intento 1: newspaper3k ──
    try:
        article = _newspaper_download(url)
        titulo  = article.title
        fecha   = (article.publish_date.strftime("%Y-%m-%d")
                   if article.publish_date else "Sin fecha")
        texto   = limpiar_texto(article.text)
        if len(texto) >= 200:
            logger.info(f"newspaper3k OK: {url}")
            return titulo, fecha, texto
        logger.warning(f"newspaper3k: texto demasiado corto ({len(texto)} chars)")
    except Exception as exc:
        logger.warning(f"newspaper3k falló para {url}: {exc}")

    # ── Intento 2: trafilatura ──
    try:
        downloaded = trafilatura.fetch_url(url)
        raw        = trafilatura.extract(downloaded, include_comments=False,
                                          include_tables=False)
        if raw and len(raw) >= 200:
            texto  = limpiar_texto(raw)
            titulo = titulo or url.split("/")[-1].replace("-", " ").title()
            logger.info(f"trafilatura OK: {url}")
            return titulo, fecha, texto
    except Exception as exc:
        logger.warning(f"trafilatura falló para {url}: {exc}")

    return None, None, None


# ─────────────────────────────────────────────
#  PROCESAMIENTO PRINCIPAL
# ─────────────────────────────────────────────
def procesar_una_url(url: str, idioma_salida: str,
                     progress_cb=None) -> str:
    """Analiza una URL y devuelve markdown con el resultado."""

    def step(pct: float, msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            progress_cb(pct, msg)

    # ── Caché ──
    fila_cache = buscar_en_cache(url)
    if fila_cache is not None:
        step(1.0, f"⚡ Resultado en caché: {url}")
        return construir_salida(fila_cache, idioma_salida, cache=True)

    step(0.05, "📥 Descargando artículo…")
    titulo, fecha, texto = extraer_texto_articulo(url)
    if not texto:
        return f"❌ No se pudo extraer texto de: `{url}`"

    step(0.15, "🌍 Detectando idioma…")
    idioma_original = detect(texto)

    step(0.30, "✂️ Generando resumen…")
    texto_truncado = truncar_por_tokens(texto, summarizer_tokenizer)
    resumen = summarizer(
        texto_truncado, max_length=150, min_length=40, do_sample=False,
    )[0]["summary_text"]

    step(0.50, "💬 Analizando sentimiento…")
    sentimiento = analizar_sentimiento(resumen)

    step(0.60, "🏷️ Clasificando tema…")
    tema = clasificar_tema(resumen)

    step(0.70, "🔍 Detectando sesgo mediático…")
    sesgo = detectar_sesgo(texto)

    step(0.80, "👤 Extrayendo entidades…")
    entidades = extraer_entidades(texto)

    step(0.90, "⭐ Calculando credibilidad…")
    credibilidad = calcular_credibilidad(texto, fecha, url)

    entry = {
        "fecha_registro": datetime.datetime.now().isoformat(),
        "url":            url,
        "titulo":         titulo,
        "fecha_articulo": fecha,
        "idioma":         idioma_original,
        "tema":           tema,
        "resumen":        resumen,
        "sentimiento":    sentimiento,
        "sesgo":          sesgo,
        "credibilidad":   credibilidad,
        "entidades":      entidades,
    }

    guardar_en_db(entry)
    step(1.0, "✅ Guardado en base de datos.")
    return construir_salida(entry, idioma_salida)


def construir_salida(fila: dict, idioma_salida: str, cache: bool = False) -> str:
    resumen     = fila["resumen"]
    idioma_orig = fila["idioma"]

    if idioma_salida != idioma_orig:
        resumen = traducir(resumen, idioma_orig, idioma_salida)

    entidades = fila.get("entidades", {})
    if isinstance(entidades, str):
        entidades = json.loads(entidades)

    personas = ", ".join(entidades.get("PER", [])) or "—"
    orgs     = ", ".join(entidades.get("ORG", [])) or "—"
    lugares  = ", ".join(entidades.get("LOC", [])) or "—"
    cred     = fila.get("credibilidad", "—")
    sesgo    = fila.get("sesgo", "—")

    salida = (
        f"📰 **Título:** {fila['titulo']}\n"
        f"📅 **Fecha del artículo:** {fila['fecha_articulo']}\n"
        f"🌍 **Idioma original:** {idioma_orig}\n"
        f"🏷️ **Tema:** {fila['tema']}\n"
        f"⚖️ **Sesgo detectado:** {sesgo}\n"
        f"⭐ **Credibilidad:** {cred} / 100\n\n"
        f"🔎 **Resumen:**\n{resumen}\n\n"
        f"💬 **Sentimiento:** {fila['sentimiento']}\n\n"
        f"**Entidades detectadas:**\n"
        f"  👤 Personas: {personas}\n"
        f"  🏢 Organizaciones: {orgs}\n"
        f"  📍 Lugares: {lugares}"
    )

    if cache:
        salida += "\n\n⚡ *Resultado recuperado de caché*"

    return salida


# ─────────────────────────────────────────────
#  EXPORTACIÓN
# ─────────────────────────────────────────────
def exportar_csv() -> str | None:
    df = obtener_historial()
    if df.empty:
        return None
    path = "/tmp/historial_noticias.csv"
    df.to_csv(path, index=False)
    return path


def exportar_json() -> str | None:
    df = obtener_historial()
    if df.empty:
        return None
    path = "/tmp/historial_noticias.json"
    df.to_json(path, orient="records", force_ascii=False, indent=2)
    return path



# ─────────────────────────────────────────────
#  ESTADÍSTICAS
# ─────────────────────────────────────────────
def generar_estadisticas():
    df = obtener_historial()
    if df.empty:
        return None, None, None

    # ── Pie: distribución de temas ──
    fig_temas = px.pie(
        df, names="tema",
        title="Distribución por Tema",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_temas.update_layout(template="plotly_dark")

    # ── Line: evolución del sentimiento ──
    df_sent = df.copy()
    df_sent["fecha_registro"] = pd.to_datetime(df_sent["fecha_registro"])
    df_sent["sent_num"] = df_sent["sentimiento"].map(SENTIMIENTO_MAP)
    df_sent = df_sent.sort_values("fecha_registro")

    fig_sent = px.line(
        df_sent, x="fecha_registro", y="sent_num",
        title="Evolución del Sentimiento",
        markers=True,
        labels={"sent_num": "Sentimiento", "fecha_registro": "Fecha"},
    )
    fig_sent.update_yaxes(
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["Muy neg.", "Negativo", "Neutral", "Positivo", "Muy pos."],
    )
    fig_sent.update_layout(template="plotly_dark")

    # ── Bar: dominios más analizados ──
    df_dom = df.copy()
    df_dom["dominio"] = df_dom["url"].apply(
        lambda x: urlparse(x).netloc.replace("www.", "")
    )
    dom_counts = (
        df_dom["dominio"].value_counts().head(10)
        .reset_index()
    )
    dom_counts.columns = ["dominio", "count"]
    fig_dom = px.bar(
        dom_counts, x="dominio", y="count",
        title="Top 10 Dominios Analizados",
        labels={"count": "Noticias", "dominio": "Dominio"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig_dom.update_layout(template="plotly_dark", showlegend=False)

    return fig_temas, fig_sent, fig_dom


# ─────────────────────────────────────────────
#  HANDLERS DE INTERFAZ
# ─────────────────────────────────────────────
def handle_individual(url: str, idioma_salida: str, progress=gr.Progress()):
    progress(0.0, desc="Iniciando análisis…")
    resultado = procesar_una_url(
        url.strip(), idioma_salida,
        progress_cb=lambda pct, msg: progress(pct, desc=msg),
    )
    progress(1.0, desc="✅ Completado")
    return resultado, exportar_csv(), exportar_json()


def handle_lote(urls_text: str, idioma_salida: str, progress=gr.Progress()):
    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    if not urls:
        return "⚠️ No se introdujeron URLs.", None, None, None

    total      = len(urls)
    resultados = [""] * total
    progress(0.0, desc=f"Procesando {total} URL(s)…")

    with ThreadPoolExecutor(max_workers=MAX_BATCH_WORKERS) as executor:
        futures = {
            executor.submit(procesar_una_url, url, idioma_salida): (i, url)
            for i, url in enumerate(urls)
        }
        completados = 0
        for future in as_completed(futures):
            i, url = futures[future]
            completados += 1
            try:
                res = future.result()
            except Exception as exc:
                res = f"❌ Error procesando `{url}`: {exc}"
            resultados[i] = f"---\n🔗 **URL {i+1}:** `{url}`\n\n{res}"
            progress(completados / total, desc=f"Completado {completados}/{total}")

    salida_final = "\n\n".join(resultados)
    return salida_final, exportar_csv(), exportar_json()


def handle_comparativa(u1: str, u2: str, u3: str,
                       idioma: str, progress=gr.Progress()):
    urls = [u.strip() for u in [u1, u2, u3] if u.strip()]
    if len(urls) < 2:
        return "⚠️ Introduce al menos 2 URLs."

    total = len(urls)
    for i, url in enumerate(urls):
        progress(i / total, desc=f"Analizando fuente {i+1}/{total}…")
        procesar_una_url(url, idioma)   # guarda en DB si no existe

    progress(0.95, desc="Generando comparativa…")

    filas = []
    conn  = sqlite3.connect(DB_PATH)
    for url in urls:
        cur = conn.execute(
            "SELECT titulo, resumen, sentimiento, sesgo, tema FROM noticias WHERE url = ?",
            (url,),
        )
        row = cur.fetchone()
        if row:
            filas.append(dict(zip(
                ["titulo", "resumen", "sentimiento", "sesgo", "tema"], row
            )))
    conn.close()

    if not filas:
        return "❌ No se pudo recuperar información de las URLs indicadas."

    salida = "## ⚖️ Comparativa de Fuentes\n\n"
    for i, (url, datos) in enumerate(zip(urls, filas), 1):
        salida += (
            f"### 📰 Fuente {i}: {datos['titulo']}\n"
            f"- **Sentimiento:** {datos['sentimiento']} | "
            f"**Sesgo:** {datos['sesgo']} | "
            f"**Tema:** {datos['tema']}\n"
            f"- **Resumen:** {datos['resumen']}\n\n"
        )

    temas      = [d["tema"] for d in filas]
    sentiments = [d["sentimiento"] for d in filas]
    sesgos     = [d["sesgo"] for d in filas]

    salida += "---\n### 📊 Análisis Comparativo\n"
    salida += f"- **Temas detectados:** {', '.join(set(temas))}\n"
    salida += f"- **Sentimientos:** {', '.join(sentiments)}\n"
    salida += f"- **Sesgos detectados:** {', '.join(set(sesgos))}\n\n"

    if len(set(temas)) == 1:
        salida += "✅ Todas las fuentes tratan el **mismo tema**.\n"
    else:
        salida += "⚠️ Las fuentes **difieren** en la clasificación temática.\n"

    if len(set(sentiments)) > 1:
        salida += "⚠️ Las fuentes presentan **tonos emocionales distintos** sobre el mismo evento.\n"
    else:
        salida += "✅ Todas las fuentes coinciden en el **tono emocional**.\n"

    if len(set(sesgos)) > 1:
        salida += "⚠️ Se detectan **diferentes tipos de sesgo** entre las fuentes.\n"

    progress(1.0, desc="✅ Comparativa lista")
    return salida


# ─────────────────────────────────────────────
#  INTERFAZ GRADIO
# ─────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="🧠 Analizador IA de Noticias") as demo:

    gr.Markdown(
        "# 🧠 Analizador IA de Noticias\n"
        "Resumen automático · Sentimiento · Entidades · Sesgo · Credibilidad · Estadísticas"
    )

    with gr.Tabs():

        # ── TAB 1: Análisis Individual ──────────────────────────
        with gr.TabItem("📰 Análisis Individual"):
            with gr.Row():
                url_input    = gr.Textbox(
                    lines=2, label="URL de la noticia",
                    placeholder="https://ejemplo.com/articulo",
                )
                idioma_radio = gr.Radio(
                    ["es", "en"], label="Idioma de salida", value="es"
                )

            btn_analizar = gr.Button("🔍 Analizar", variant="primary")
            resultado_md = gr.Markdown(label="Resultado del análisis")

            with gr.Row():
                out_csv  = gr.File(label="📄 Historial CSV")
                out_json = gr.File(label="📦 Historial JSON")

            btn_analizar.click(
                fn=handle_individual,
                inputs=[url_input, idioma_radio],
                outputs=[resultado_md, out_csv, out_json],
            )

        # ── TAB 2: Análisis en Lote ──────────────────────────────
        with gr.TabItem("📋 Análisis en Lote"):
            gr.Markdown(
                f"Pega una URL por línea. Se procesan en paralelo "
                f"(máx. {MAX_BATCH_WORKERS} a la vez)."
            )
            urls_batch   = gr.Textbox(
                lines=6, label="URLs (una por línea)",
                placeholder="https://...\nhttps://...",
            )
            idioma_lote  = gr.Radio(["es", "en"], label="Idioma de salida", value="es")
            btn_lote     = gr.Button("🚀 Procesar lote", variant="primary")
            resultado_lote = gr.Markdown()

            with gr.Row():
                lote_csv  = gr.File(label="📄 CSV")
                lote_json = gr.File(label="📦 JSON")

            btn_lote.click(
                fn=handle_lote,
                inputs=[urls_batch, idioma_lote],
                outputs=[resultado_lote, lote_csv, lote_json],
            )

        # ── TAB 3: Comparativa Multi-URL ─────────────────────────
        with gr.TabItem("⚖️ Comparativa"):
            gr.Markdown(
                "Analiza 2–3 artículos sobre el mismo tema y detecta "
                "coincidencias, diferencias de tono y sesgo entre fuentes."
            )
            with gr.Row():
                url_c1 = gr.Textbox(label="URL 1", placeholder="https://…")
                url_c2 = gr.Textbox(label="URL 2", placeholder="https://…")
                url_c3 = gr.Textbox(label="URL 3 (opcional)", placeholder="https://…")

            idioma_comp    = gr.Radio(["es", "en"], label="Idioma de salida", value="es")
            btn_comp       = gr.Button("⚖️ Comparar fuentes", variant="primary")
            resultado_comp = gr.Markdown()

            btn_comp.click(
                fn=handle_comparativa,
                inputs=[url_c1, url_c2, url_c3, idioma_comp],
                outputs=[resultado_comp],
            )

        # ── TAB 4: Historial ─────────────────────────────────────
        with gr.TabItem("📚 Historial"):
            btn_refresh  = gr.Button("🔄 Actualizar historial")
            historial_df = gr.Dataframe(
                headers=["fecha_registro","titulo","tema","sentimiento",
                         "sesgo","credibilidad","url"],
                label="Noticias analizadas",
                interactive=False,
                wrap=True,
            )
            btn_refresh.click(fn=obtener_historial, outputs=[historial_df])
            demo.load(fn=obtener_historial, outputs=[historial_df])

        # ── TAB 5: Estadísticas ──────────────────────────────────
        with gr.TabItem("📊 Estadísticas"):
            btn_stats = gr.Button("📈 Generar gráficas")

            with gr.Row():
                plot_temas = gr.Plot(label="Distribución por Tema")
                plot_sent  = gr.Plot(label="Evolución del Sentimiento")

            plot_dom = gr.Plot(label="Top Dominios Analizados")

            btn_stats.click(
                fn=generar_estadisticas,
                outputs=[plot_temas, plot_sent, plot_dom],
            )

# ─────────────────────────────────────────────
#  ARRANQUE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Iniciando aplicación Gradio…")
    demo.launch()
