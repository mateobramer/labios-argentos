# README DATASET

## 1. Que es este bucket

Bucket principal:

`gs://labios-argentos-vsr-clean-v1/`

Este bucket contiene un dataset de VSR / lectura de labios para español argentino
rioplatense. Combina dos bloques:

- datos existentes reconstruidos desde el dataset previo (`argentina/existing`);
- clips nuevos recolectados en la etapa `new_discovery` (`argentina/new_discovery`).

El objetivo es tener clips cortos con video, transcripcion y, cuando exista, ROI labial
listo para entrenamiento o evaluacion de modelos de lectura de labios.

## 2. Estructura principal

- Viejos / existing:
  `gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4/`

- Nuevos / new_discovery:
  `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/`

- Transcripciones new_discovery:
  `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/transcripts/`

- ROIs new_discovery:
  `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/rois_npz/`

- Manifests combinados:
  `gs://labios-argentos-vsr-clean-v1/argentina/combined/manifests/`

- Manifests generales:
  `gs://labios-argentos-vsr-clean-v1/manifests/`

- Reportes:
  `gs://labios-argentos-vsr-clean-v1/reports/`

## 3. Que manifest usar

- Para ver todo: `manifests/final_release_manifest.csv`.
- Para entrenamiento conservador VSR: usar filas con `usable_for_training=true` y
  `npz_path` no vacio.
- Para texto mas limpio: usar `selected_training_text`; si `gpt_status=completed_clean_gpt`,
  ese texto viene de `clean_gpt_v1`.
- Para inspeccionar GPT: `manifests/clean_gpt_manifest.csv`.
- Para ROI: `manifests/new_discovery_roi_manifest.csv` y
  `reports/new_discovery_roi_report.md`.

## 4. Prioridad de transcripciones

Regla recomendada:

1. Usar `clean_gpt_v1` si `gpt_status=completed_clean_gpt`.
2. Usar ASR `large` / `turbo` si no hay GPT.
3. Usar baseline existing si no hay `large` / `turbo`.
4. No usar texto `blocked` o `needs_review` sin revision manual.

`clean_gpt` no significa texto idealizado. La limpieza GPT es conservadora: no debe
inventar frases, no debe borrar disfluencias reales y no debe cambiar el registro oral
cuando la evidencia de ASR no alcanza.

## 5. Conteos finales

Conteos despues de aplicar los outputs GPT manuales disponibles:

- `final_release_manifest.csv`: 22.384 filas.
- Existing: 9.191 filas.
- New discovery: 13.193 filas.
- `large` completo: 15.327 clips.
- `large+turbo` completo: 15.327 clips.
- GPT completed (`gpt_status=completed_clean_gpt`): 13.190 clips.
- GPT rejected (`gpt_status=rejected_clean_gpt`): 1.613 clips.
- GPT validation rejected: 26 outputs.
- GPT pending/missing: 500 clips en 3 raw outputs faltantes.
- ROI valido (`npz_path` no vacio): 11.439 filas.
- ROI faltante: 10.945 filas.
- `usable_for_training=true`: 11.439 filas.

## 6. Estado de ROI

ROI recovery fue dado de baja para esta entrega. El ROI valido actual de
`new_discovery` era 2.248 antes de cualquier cambio posterior, y se conserva ese estado:
muchos clips `new_discovery` tienen audio/transcripcion pero no ROI valido.

`blocked_no_face` / `blocked_roi_no_face` no necesariamente significa que no haya cara
en el video. Puede ser un problema de decode, preprocesamiento, threshold o MediaPipe.
La recuperacion de ROI queda delegada a otra persona.

No relanzar VM a ciegas. Primero diagnosticar decode/AV1/reencode con una muestra:
`ffprobe`, revisar codec, reencodear una muestra a H264 y recien despues reintentar ROI
en sample antes de correr una VM completa.

## 7. Estado de GPT cleaning

No se uso GPT automatico, browser ni API para el full. Se preparo un handoff manual y se
ingirieron los outputs JSONL entregados en ZIP.

Los outputs manuales fueron validados contra
`cleaning/gpt_clean_v1/manual_gpt_handoff/job_index.csv`. Solo se aplicaron jobs con
status `validated`. Los jobs faltantes, invalidos o incompletos quedan rechazados o
pendientes en:

- `cleaning/gpt_clean_v1/reports/manual_gpt_validation_report.csv`
- `cleaning/gpt_clean_v1/reports/manual_gpt_apply_report.md`
- `cleaning/gpt_clean_v1/rejected_patches.jsonl`

## 8. Como reproducir validacion

Desde la raiz del repo:

```powershell
python -m compileall cleaning/gpt_clean_v1 data_pipeline/release data_pipeline/discovery
python -m unittest discover data_pipeline/discovery\tests
git diff --check
python data_pipeline/release/scripts/validate_clean_bucket.py
```

Si `validate_clean_bucket.py` tarda demasiado, usar una validacion equivalente que
compare manifests locales contra los paths subidos y deje evidencia en
`data_pipeline/release/reports/`.

## 9. Advertencias

- No asumir que todo clip con transcripcion sirve para VSR.
- Para VSR usar `usable_for_training=true` y ROI valido.
- Para NLP/texto se puede usar texto limpio aunque no haya ROI.
- No tocar originales.
- No borrar manifests viejos sin versionar.
- No subir outputs GPT sin validar.
- Cualquier cambio futuro en bucket/manifests debe actualizar `README_DATASET.md` y
  `OPEN_ITEMS_DATASET.md`.
