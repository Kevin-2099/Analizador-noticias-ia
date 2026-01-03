import gradio as gr
from newspaper import Article
from transformers import pipeline, MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import datetime
import pandas as pd
import torch
import re

# ============== MODELOS ==============
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_tokenizer = summarizer.tokenizer

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

topic_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)

TOPICS = [
    "política y gobierno",
    "economía y finanzas",
    "tecnología e innovación",
    "salud y medicina",
    "deportes",
]

# Traductores
en_to_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_to_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

es_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
es_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

CSV_PATH = "logs_resumenes.csv"

# Inicializar logs
try:
    log_df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    log_df = pd.DataFrame(
        columns=[
            "fecha_registro",
            "url",
            "titulo",
            "fecha_articulo",
            "idioma",
            "tema",
            "resumen",
            "sentimiento",
        ]
    )

# ============== FUNCIONES ==============

def limpiar_texto(texto):
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(
        r"(suscríbete|subscribe|cookies|privacy policy)",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    return texto.strip()


def truncar_por_tokens(texto, tokenizer, max_tokens=1024):
    tokens = tokenizer(
        texto, truncation=True, max_length=max_tokens, return_tensors="pt"
    )
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)


def traducir(texto, origen, destino):
    if origen == destino:
        return texto

    if origen == "en" and destino == "es":
        tokenizer, model = en_to_es_tokenizer, en_to_es_model
    elif origen == "es" and destino == "en":
        tokenizer, model = es_to_en_tokenizer, es_to_en_model
    else:
        return texto

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def analizar_sentimiento(texto):
    inputs = sentiment_tokenizer(
        texto, return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    estrellas = int(torch.argmax(outputs.logits)) + 1
    return ["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"][estrellas - 1]


def clasificar_tema(texto):
    result = topic_classifier(texto, TOPICS)
    label = result["labels"][0].lower()

    if "política" in label:
        return "Política"
    if "economía" in label:
        return "Economía"
    if "tecnología" in label:
        return "Tecnología"
    if "salud" in label:
        return "Salud"
    if "deportes" in label:
        return "Deportes"
    return "Otros"


def buscar_en_cache(url):
    resultado = log_df[log_df["url"] == url]
    if not resultado.empty:
        return resultado.iloc[0]
    return None


def construir_salida(fila, idioma_salida, cache=False):
    resumen = fila["resumen"]
    sentimiento = fila["sentimiento"]

    if idioma_salida != fila["idioma"]:
        resumen = traducir(resumen, fila["idioma"], idioma_salida)
        sentimiento = traducir(sentimiento, fila["idioma"], idioma_salida)

    salida = (
        f"📰 **Título:** {fila['titulo']}\n"
        f"📅 **Fecha:** {fila['fecha_articulo']}\n"
        f"🌍 **Idioma original:** {fila['idioma']}\n"
        f"🏷️ **Tema:** {fila['tema']}\n\n"
        f"🔎 **Resumen:**\n{resumen}\n\n"
        f"💬 **Sentimiento:** {sentimiento}"
    )

    if cache:
        salida += "\n\n⚡ *Resultado recuperado de cache*"

    return salida


def procesar_noticia(url, idioma_salida):
    global log_df

    # 🔹 CACHE
    fila_cache = buscar_en_cache(url)
    if fila_cache is not None:
        return construir_salida(fila_cache, idioma_salida, cache=True), CSV_PATH

    try:
        article = Article(url)
        article.download()
        article.parse()

        titulo = article.title
        fecha = (
            article.publish_date.strftime("%Y-%m-%d")
            if article.publish_date
            else "Sin fecha"
        )

        texto = limpiar_texto(article.text)
        if len(texto) < 200:
            return "❌ Artículo demasiado corto.", None

        idioma_original = detect(texto)

        texto_truncado = truncar_por_tokens(texto, summarizer_tokenizer)
        resumen = summarizer(
            texto_truncado, max_length=150, min_length=40, do_sample=False
        )[0]["summary_text"]

        sentimiento = analizar_sentimiento(resumen)
        tema = clasificar_tema(resumen)

        log_entry = {
            "fecha_registro": datetime.datetime.now().isoformat(),
            "url": url,
            "titulo": titulo,
            "fecha_articulo": fecha,
            "idioma": idioma_original,
            "tema": tema,
            "resumen": resumen,
            "sentimiento": sentimiento,
        }

        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        log_df.to_csv(CSV_PATH, index=False)

        return construir_salida(log_entry, idioma_salida), CSV_PATH

    except Exception as e:
        return f"❌ Error: {e}", None


# ============== INTERFAZ ==============

demo = gr.Interface(
    fn=procesar_noticia,
    inputs=[
        gr.Textbox(lines=2, label="URL de noticia"),
        gr.Radio(["es", "en"], label="Idioma de salida", value="es"),
    ],
    outputs=[
        gr.Markdown(label="Resultado"),
        gr.File(label="Descargar CSV"),
    ],
    title="🧠 Analizador IA de Noticias",
    description=(
        "Resumen automático, análisis de sentimiento, clasificación temática "
        "y cache inteligente por URL. El idioma de salida es siempre configurable."
    ),
)

demo.launch()
