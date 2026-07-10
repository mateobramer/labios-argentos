# data/ — qué hay acá y cómo recuperar lo que se retiró

En la reorganización 2026-07 (branch `chore/repo-cleanup-safe-v2`) los datos masivos
se **retiraron del árbol de esta rama** (siguen intactos en `main`, en el tag
`dataset-clean-v1` y en los buckets). Política completa:
[`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md).

## Qué queda versionado acá

| Path | Qué es | Tamaño |
|---|---|---|
| `corpus/` | transcripciones Whisper por fuente (cache, texto) | ~19 MB |
| `metadata/` | manifests chicos: `fuentes.csv`, `lip_preprocessing_manifest.csv`, `auditoria_clips_manifest.csv`, muestras vsr_eval | ~2.5 MB |
| `samples/El mensaje de Coscu sobre EL AMOR/` | **muestra mínima para smoke tests**: 8 clips mp4+txt | 2.9 MB |

## Qué se retiró de esta rama (manifest de recuperación)

| Retirado | Contenido | Recuperación exacta |
|---|---|---|
| `data/clips/` (17.078 archivos, ~2.24 GB) | clips alineados mp4+txt, rondas 1-2 | `git checkout main -- "data/clips"` (bytes exactos) o bucket |
| `dataset/` (11.901 archivos, ~248 MB) | clips `keep` post-curación | `git checkout main -- dataset` |
| `data/videos/` (9 archivos, ~422 MB) | videos fuente crudos | `git checkout main -- data/videos` o regenerar: `python data_pipeline/descargar_procesar.py <url>` |
| `data/metadata/visual_quality_{policy_*,*_full_roi_sanity}.csv` (6 archivos, ~54 MB) | outputs del análisis de calidad visual | `git checkout main -- data/metadata` (trae también los chicos; los 6 grandes quedan ignorados por .gitignore, usar `git checkout main -- "data/metadata/<nombre>.csv"` puntual) |

También todo está congelado en el tag: `git show dataset-clean-v1:<path>`.

### Desde el bucket (forma release, requiere credenciales del proyecto GCP)

```bash
# clips existing (12.112 mp4 según bucket_validation_report.md):
gsutil -m cp -r "gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4" ./
# nuevos + transcripts + ROIs:
gsutil -m cp -r "gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio" ./
# manifest maestro:
gsutil cp "gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv" ./
```

Guía completa del bucket: [`data_pipeline/release/reports/HOW_TO_USE_BUCKET.md`](../data_pipeline/release/reports/HOW_TO_USE_BUCKET.md).
Conteos validados por el equipo: [`data_pipeline/release/reports/bucket_validation_report.md`](../data_pipeline/release/reports/bucket_validation_report.md).

## Por qué la muestra es esa

`El mensaje de Coscu sobre EL AMOR` es la fuente más chica del dataset (8 clips,
2.9 MB): alcanza para smoke tests de `preprocessing/` y del pipeline sin inflar el
clone. No es representativa para evaluación — para eso están los splits congelados
(`vsr/splits/`) sobre el dataset completo.
