# claude-videos — selección de fuentes del dataset

Listas curadas de videos de YouTube que entraron al dataset VSR rioplatense. Este es el
*gate 0* (selección) del pipeline de datos: se elige y verifica **antes** de gastar
descarga/transcripción/GPU. Las dos rondas ya fueron procesadas.

## Qué hay acá

- `candidatos.csv` — **ronda 1** (39 videos, ~10.7 h, 15 hablantes). Verificados con
  `yt-dlp`. Con estos se entrenaron ft03/ft04.
- `candidatos_v2_FINAL.csv` — **ronda 2** (32 fuentes seleccionadas tras pasada visual).
  Con estos (ronda 1 + 2) se entrenaron ft05/ft06 (~19 h totales de train junto a LIP-RTVE).
- `candidatos_v2_FINAL_urls.txt` — las URLs de la ronda 2, una por línea.

## Criterios de selección (ambas rondas)

- **Un solo hablante dominante a cámara.** Se descartó todo lo multi-hablante (podcasts
  mano a mano, mesas): el habla solapada ensucia la transcripción de Whisper y rompe la
  invariante clip↔`.txt` (el preproc recorta una sola boca por clip).
- **Informal y bien argentino** (slang, voseo, muletillas), no discurso formal.
- **Rioplatense** (Buenos Aires / GBA / bonaerense). Sin acento del interior.
- La **frontalidad real de la boca** no se puede confirmar sin mirar el video; la decide
  el pipeline (gate de alineación + descarte por cara <80% en el preproc visual). Por eso
  las listas sobre-aprovisionan horas.

## Cómo se procesa una fuente nueva

```
python descargar_procesar.py "URL"                                  # etapa 1 + gate alineación
python -m preprocessing.src.preprocesar "<titulo>"           # etapa 2 (ROIs 96×96)
python -m cleaning.visual_quality.src.detectar_clips_malos "<titulo>"         # etapa 3 (review)
python -m cleaning.visual_quality.src.detectar_clips_malos "<titulo>" --materializar
```

Al final se completa la fila en `data/metadata/fuentes.csv`. Ver el flujo completo en
[`docs/ESTRUCTURA.md`](../docs/ESTRUCTURA.md). No editar a mano los datos generados
bajo `data/`.
