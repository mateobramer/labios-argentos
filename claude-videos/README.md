# claude-videos — candidatos para sumar al dataset

Lista curada de videos de YouTube **candidatos** a entrar al dataset VSR rioplatense.
**No hay nada descargado todavía**: esto es el *gate 0* (selección) de
`PIPELINE_PROYECTO.md`, hecho antes de gastar descarga/transcripción/GPU.

## Qué hay acá

- `candidatos.csv` — 39 videos, ~10.7 h, **todos verificados con `yt-dlp`** (existen,
  públicos, duración real). Son **monólogo / storytime a cámara de un solo hablante**
  rioplatense, registro **informal coloquial** (slang, voseo, muletillas). 15 hablantes
  distintos, balance de género razonable (8 varones / 7 mujeres).

## Criterios con que se eligieron

- **Un solo hablante dominante a cámara.** Decisión deliberada: se descartó todo lo
  multi-hablante (podcasts mano a mano, mesas) porque el habla solapada ensucia la
  transcripción de Whisper y rompe la invariante clip↔`.txt` (el preproc recorta una sola
  boca por clip).
- **Informal y bien argentino**, no discurso formal/correcto.
- **Rioplatense** (Buenos Aires / GBA / bonaerense). Sin acento del interior.

## Lo que NO se pudo verificar todavía (importante)

La **frontalidad real de la boca** no se puede confirmar sin mirar el video; eso lo decide
el pipeline (gate de alineación + descarte por cara <80% en el preproc visual). La columna
`frontalidad_nota` es una estimación por formato/frames: lo marcado como "revisar cortes en
gate" o con caveat (micrófono, canal mixto) hay que mirarlo con más cuidado en el gate 3.
**Es esperable perder material en el preproc**, por eso la lista sobre-aprovisiona un poco
arriba de 10 h.

## Cómo seguir (con tu OK)

Cada fila se procesa con el skill `/nueva-fuente`, que orquesta el pipeline existente con
los gates de calidad:

```
python descargar_procesar.py "URL"                                  # etapa 1 + gate alineacion
python -m visual_preprocessing.src.preprocesar "<titulo>"           # etapa 2
python -m data_cleaning.src.detectar_clips_malos "<titulo>"         # etapa 3 (review)
python -m data_cleaning.src.detectar_clips_malos "<titulo>" --materializar
```

Recién al final se completa la fila en `data/metadata/fuentes.csv` (etapa 4). Este CSV es
solo la antesala; no editar a mano los datos generados bajo `data/` ni `dataset/`.
