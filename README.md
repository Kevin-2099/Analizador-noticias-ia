# Analizador-noticias-ia
Este proyecto es una aplicación interactiva construida con Gradio que permite analizar automáticamente artículos de noticias desde una URL. El sistema hace scraping del contenido, resume el texto, detecta el idioma, analiza el sentimiento del contenido y guarda un registro de los resultados.

## 🚀 Funcionalidades
🌐 Extracción automática de contenido de artículos a partir de la URL usando newspaper3k.

🧾 Resumen automático del texto con modelos BART de Hugging Face.

🌍 Detección de idioma usando langdetect.

🔄 Traducción automática entre inglés y español con modelos MarianMT.

💬 Análisis de sentimiento con BERT multilingüe.

🧾 Registro automático de resultados en un archivo CSV.

## 🛠️ Tecnologías utilizadas
Gradio

newspaper3k

transformers

langdetect

torch

pandas

## 📦 Instalación
.Clona el repositorio:

git clone https://github.com/tu-usuario/analizador-ia-noticias.git

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
Introduce la URL de una noticia.

Espera unos segundos mientras el sistema procesa.

Obtendrás:

Título y fecha del artículo

Idioma detectado

Resumen generado

Sentimiento clasificado (Muy negativo → Muy positivo)

## 📄 Licencia
Este proyecto se distribuye bajo la licencia MIT. Ver archivo LICENSE para más detalles.

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Puedes:

Reportar errores

Sugerir mejoras

Hacer pull requests
