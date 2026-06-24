# Artefactos finales del analisis Azzaro

Esta carpeta contiene los artefactos chicos necesarios para reproducir y revisar el
notebook `data_cleaning/notebooks/02_analisis_whisper_azzaro.ipynb`
sin volver a ejecutar Whisper ni descargar el video original.

Contenido esperado:

- `transcripts/`: tres JSON generados con MLX Whisper (`turbo`, `small` y `large`),
  todos con timestamps por palabra.
- `clips/`: solo los once clips WebM seleccionados manualmente para el informe.
- `seleccion.csv`: tiempos y transcripciones comparadas de esos once casos.

El video original de YouTube, los modelos descargados y los clips descartados no se
versionan.
