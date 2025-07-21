import gradio as gr
from newspaper import Article
from transformers import pipeline, MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import datetime
import pandas as pd
import torch

# ============== MODELOS ==============
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Traductores
en_to_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_to_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

es_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
es_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

# CSV path
CSV_PATH = "logs_resumenes.csv"

# Inicializar logs si no existe
try:
    log_df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["fecha_registro", "url", "titulo", "fecha_articulo", "idioma", "resumen", "sentimiento"])

# ============== FUNCIONES ==============

def traducir(texto, origen='en', destino='es'):
    if origen == 'en' and destino == 'es':
        tokenizer, model = en_to_es_tokenizer, en_to_es_model
    elif origen == 'es' and destino == 'en':
        tokenizer, model = es_to_en_tokenizer, es_to_en_model
    else:
        return texto

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def analizar_sentimiento(texto):
    inputs = sentiment_tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = outputs.logits.softmax(dim=1).squeeze()
    estrellas = int(torch.argmax(scores)) + 1
    sentimiento = ["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"][estrellas - 1]
    return sentimiento

def procesar_noticia(url):
    global log_df
    try:
        article = Article(url)
        article.download()
        article.parse()

        titulo = article.title
        fecha = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else "Sin fecha"
        texto = article.text

        if len(texto) < 200:
            return "❌ El artículo es demasiado corto para analizarlo.", None

        idioma = detect(texto)

        texto_resumen = texto
        if idioma == 'en':
            texto_resumen = traducir(texto, origen='en', destino='es')

        if len(texto_resumen) > 1024:
            texto_resumen = texto_resumen[:1024]

        resumen = summarizer(texto_resumen, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        sentimiento = analizar_sentimiento(texto_resumen)

        # Guardar log
        log_entry = {
            "fecha_registro": datetime.datetime.now().isoformat(),
            "url": url,
            "titulo": titulo,
            "fecha_articulo": fecha,
            "idioma": idioma,
            "resumen": resumen,
            "sentimiento": sentimiento
        }

        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        log_df.to_csv(CSV_PATH, index=False)

        # Preparar salida
        if idioma == 'en':
            resumen = traducir(resumen, origen='es', destino='en')
            sentimiento = traducir(sentimiento, origen='es', destino='en')
            salida = f"📰 **Title:** {titulo}\n📅 **Date:** {fecha}\n🌍 **Detected language:** {idioma}\n\n🔎 **Summary:**\n{resumen}\n\n💬 **Sentiment:** {sentimiento}"
        else:
            salida = f"📰 **Título:** {titulo}\n📅 **Fecha:** {fecha}\n🌍 **Idioma detectado:** {idioma}\n\n🔎 **Resumen:**\n{resumen}\n\n💬 **Sentimiento:** {sentimiento}"

        return salida, CSV_PATH

    except Exception as e:
        return f"❌ Error al procesar la noticia: {e}", None

# ============== INTERFAZ GRADIO ==============

demo = gr.Interface(
    fn=procesar_noticia,
    inputs=gr.Textbox(lines=2, label="URL de noticia"),
    outputs=[
        gr.Markdown(label="Resultado"),
        gr.File(label="Descargar historial CSV")
    ],
    title="🧠 Analizador IA de Noticias",
    description="Introduce la URL de una noticia. El sistema hará scraping, resumirá el contenido, detectará idioma, analizará sentimiento y generará un CSV descargable con el historial.",
)

demo.launch()
