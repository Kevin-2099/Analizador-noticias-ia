## 🧠 Analizador IA de Noticias

Aplicación interactiva construida con Gradio que permite analizar automáticamente artículos de noticias a partir de una URL utilizando modelos avanzados de Procesamiento de Lenguaje Natural (NLP).

El sistema no solo resume noticias, sino que también evalúa su sentimiento, sesgo, credibilidad y entidades, ofreciendo una visión mucho más completa del contenido mediático.

## 🚀 Funcionalidades
- 📰 Análisis de contenido
- 🌐 Extracción automática de artículos desde URL
(fallback robusto con newspaper3k + trafilatura)
- ✂️ Resumen automático con modelos BART
- 🌍 Detección de idioma (es / en)
- 🔄 Traducción automática entre español e inglés
- 🧠 Análisis NLP avanzado
- 💬 Análisis de sentimiento (Muy negativo → Muy positivo)
- 🏷️ Clasificación temática automática
- ⚖️ Detección de sesgo mediático:
  - Neutral
  - Sensacionalista
  - Opinativo
  - Propagandístico
- 👤 Extracción de entidades:
  - Personas
  - Organizaciones
  - Lugares
- ⭐ Evaluación de calidad
  - Cálculo de credibilidad (0–100) basado en:
    - Longitud del contenido
    - Presencia de fuentes
    - Actualidad del artículo
    - Dominio del medio
- ⚡ Sistema inteligente
  - Cache persistente con SQLite
  - Expiración configurable de resultados
  - Evita reprocesar URLs ya analizadas
## 🧩 Funcionalidades avanzadas
- 📋 Análisis en lote
  - Procesa múltiples URLs en paralelo
  - Optimizado con multithreading
- ⚖️ Comparativa de fuentes
  - Compara 2–3 artículos sobre el mismo tema
  - Detecta diferencias en:
    - Sentimiento
    - Sesgo
    - Enfoque temático
- 📊 Estadísticas interactivas
  - Distribución por tema
  - Evolución del sentimiento
  - Top dominios analizados
- 📚 Historial persistente
  - Almacenamiento en base de datos SQLite
  - Consulta desde la interfaz
- 📁 Exportación
  - CSV
  - JSON
## 🛠️ Tecnologías utilizadas
- 🤖 NLP / IA
  - facebook/bart-large-cnn → Resumen
  - facebook/bart-large-mnli → Clasificación y sesgo (zero-shot)
  - nlptown/bert-base-multilingual-uncased-sentiment → Sentimiento
  - dslim/bert-base-NER → Entidades (NER)
  - Helsinki-NLP MarianMT → Traducción
- 📰 Procesamiento de contenido
  - newspaper3k
  - trafilatura
- ⚙️ Backend
  - Python
  - PyTorch
  - SQLite
  - ThreadPoolExecutor (paralelismo)
  - Tenacity (reintentos)
  - Loguru (logging)
- 📊 Frontend
  - Gradio
  - Plotly
📦 Instalación

Clona el repositorio:

git clone https://github.com/Kevin-2099/analizador-ia-noticias.git
cd analizador-ia-noticias

Crea un entorno virtual (opcional pero recomendado):

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Instala dependencias:

pip install -r requirements.txt

⚠️ Asegúrate de tener PyTorch instalado correctamente:
https://pytorch.org/get-started/locally/

🧪 Uso

Ejecuta la aplicación:

python app.py

Abre el navegador en:

http://127.0.0.1:7860
📌 Cómo usar
🔹 Análisis individual
Introduce una URL
Selecciona idioma de salida
Haz clic en Analizar
🔹 Análisis en lote
Introduce múltiples URLs (una por línea)
🔹 Comparativa
Introduce 2 o 3 URLs para comparar cobertura mediática
📊 Resultados obtenidos
📰 Título y fecha
🌍 Idioma original
🏷️ Tema
⚖️ Sesgo
⭐ Credibilidad
✂️ Resumen
💬 Sentimiento
👤 Entidades detectadas
⚡ Indicador de caché
📄 Licencia

Este proyecto está bajo licencia MIT.
Consulta el archivo LICENSE para más información.

🤝 Contribuciones

Las contribuciones son bienvenidas.

Puedes:

Hacer fork del repositorio
Crear una nueva rama (feature/nueva-funcionalidad)
Enviar un pull request

## 📄 Licencia
Este proyecto se distribuye bajo la licencia MIT. Ver archivo LICENSE para más detalles.

Hacer pull requests
