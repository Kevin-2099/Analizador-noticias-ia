
import gradio as gr
from newspaper import Article
from transformers import pipeline, MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import datetime
import pandas as pd
import torch
import numpy as np

# ============== MODELOS ==============
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Traductores
en_to_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_to_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

es_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
es_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

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

def guardar_log(url, titulo, fecha, idioma, resumen, sentimiento):
    log_entry = {
        "fecha_registro": datetime.datetime.now().isoformat(),
        "url": url,
        "titulo": titulo,
        "fecha_articulo": fecha,
        "idioma": idioma,
        "resumen": resumen,
        "sentimiento": sentimiento
    }

    try:
        df = pd.read_csv("logs_resumenes.csv")
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([log_entry])

    df.to_csv("logs_resumenes.csv", index=False)

def procesar_noticia(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        titulo = article.title
        fecha = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else "Sin fecha"
        texto = article.text

        if len(texto) < 200:
            return "❌ El artículo es demasiado corto para analizarlo."

        idioma = detect(texto)

        # Texto para resumen
        texto_resumen = texto
        if idioma == 'en':
            texto_resumen = traducir(texto, origen='en', destino='es')

        if len(texto_resumen) > 1024:
            texto_resumen = texto_resumen[:1024]

        resumen = summarizer(texto_resumen, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        sentimiento = analizar_sentimiento(texto_resumen)

        # Guardar log con datos originales (resumen en español)
        guardar_log(url, titulo, fecha, idioma, resumen, sentimiento)

        # Preparar la salida según idioma original
        if idioma == 'en':
            resumen = traducir(resumen, origen='es', destino='en')
            sentimiento = traducir(sentimiento, origen='es', destino='en')
            salida = f"📰 **Title:** {titulo}
📅 **Date:** {fecha}
🌍 **Detected language:** {idioma}

🔎 **Summary:**
{resumen}

💬 **Sentiment:** {sentimiento}"
        else:
            salida = f"📰 **Título:** {titulo}
📅 **Fecha:** {fecha}
🌍 **Idioma detectado:** {idioma}

🔎 **Resumen:**
{resumen}

💬 **Sentimiento:** {sentimiento}"

        return salida

    except Exception as e:
        return f"❌ Error al procesar la noticia: {e}"

# ============== INTERFAZ GRADIO ==============

iface = gr.Interface(
    fn=procesar_noticia,
    inputs=gr.Textbox(lines=2, label="URL de noticia"),
    outputs="markdown",
    title="🧠 Analizador IA de Noticias",
    description="Introduce la URL de una noticia. El sistema hará scraping, resumirá el contenido, detectará idioma, analizará sentimiento y mostrará todo.",
    theme="default"
)

iface.launch()
