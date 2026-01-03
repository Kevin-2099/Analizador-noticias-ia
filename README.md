# Analizador-noticias-ia
Este proyecto es una aplicación interactiva construida con Gradio que permite analizar automáticamente artículos de noticias desde una URL.

El sistema realiza scraping del contenido, limpia y resume el texto, detecta el idioma, analiza el sentimiento, clasifica el tema y guarda un registro de los resultados.

## 🚀 Funcionalidades
- 🌐 Extracción automática de contenido de artículos desde la URL usando newspaper3k

- ✂️ Resumen automático del texto con modelos BART de Hugging Face

- 🌍 Detección de idioma (español / inglés) usando langdetect

- 🔄 Traducción automática entre inglés y español con MarianMT

- 💬 Análisis de sentimiento multilingüe con BERT

- 🏷️ Clasificación temática automática (Política, Economía, Tecnología, Salud, Deportes u Otros) con zero-shot BART-MNLI

- ⚡ Cache inteligente por URL: evita reprocesar artículos ya analizados

- 🧾 Registro automático de resultados en un archivo CSV descargable

## 🛠️ Tecnologías utilizadas
- Gradio – Interfaz web interactiva

- newspaper3k – Extracción y limpieza de noticias

- transformers (Hugging Face) – Modelos BART, MarianMT, BERT y BART-MNLI

- langdetect – Detección de idioma

- torch – PyTorch, backend de modelos de NLP

- pandas – Gestión de logs y CSV

## 📦 Instalación
.Clona el repositorio:

git clone https://github.com/Kevin-2099/analizador-ia-noticias.git

cd analizador-ia-noticias

.Crea un entorno virtual (opcional pero recomendado):

python -m venv venv

source venv/bin/activate  # En Windows: venv\Scripts\activate

Instala las dependencias:

pip install -r requirements.txt

Nota: Asegúrate de tener instalado pytorch correctamente según tu sistema. Consulta https://pytorch.org/get-started/locally/ si necesitas ayuda.

## 🧪 Uso
Ejecuta el archivo principal para iniciar la interfaz web:

python app.py

Luego abre el navegador en la dirección que Gradio indique (por defecto: http://127.0.0.1:7860).

## Instrucciones
Introduce la URL de una noticia (en español o inglés) y selecciona el idioma de salida deseado

Espera unos segundos mientras el sistema procesa la noticia o la recupera desde cache

Obtendrás:

- 📰 Título y fecha del artículo

- 🌍 Idioma original detectado

- 🏷️ Tema clasificado automáticamente

- ✂️ Resumen generado

- 💬 Sentimiento estimado (Muy negativo → Muy positivo)

- ⚡ Indicador si el resultado proviene de cache

- 🧾 CSV con historial completo de análisis descargable

## 📄 Licencia
Este proyecto se distribuye bajo la licencia MIT. Ver archivo LICENSE para más detalles.

Hacer pull requests
