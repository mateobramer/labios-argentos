# Evaluation

Modulo para comparaciones, metricas y analisis de errores del proyecto.

## Notebooks

- `notebooks/01_analisis_whisper_turbo_vs_small_azzaro.ipynb`: compara Whisper
  `turbo` contra `small` en un video puntual. El notebook:
  - descarga o usa un video local;
  - transcribe con ambos modelos;
  - alinea palabras normalizadas;
  - lista los grupos de palabras que difieren;
  - exporta clips de video alrededor de cada diferencia para revision manual.

## Salidas locales

Las salidas van en `evaluation/outputs/` y no se versionan:

- videos descargados para el experimento;
- transcripciones cacheadas;
- CSVs de diferencias;
- clips de revision.

