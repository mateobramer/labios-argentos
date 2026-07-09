# Repo map

Mapa operativo despues del cleanup local de 2026-07-09.

## Bucket final

Dataset final:

```text
gs://labios-argentos-vsr-clean-v1/
```

Abrir primero:

```text
gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv
```

Para entrenamiento VSR conservador:

- usar `data_release/final_train_manifest_clean_gpt_v1.csv`;
- filtrar `usable_for_training=true`;
- exigir `npz_path` no vacio;
- usar `selected_training_text` como texto recomendado.

## Carpetas activas locales

| Path | Uso |
| --- | --- |
| `data_release/` | Manifests, reportes y scripts del release final. |
| `data_release/scripts/` | Builders y validadores del release. |
| `data_release/reports/` | Reportes finales y guia de uso del bucket. |
| `data_cleaning/` | Codigo de auditoria/curacion historica y reutilizable. |
| `data_cleaning_clean_v1/` | Codigo y reportes de limpieza GPT/manual; sin raw outputs locales. |
| `data_discovery/` | Codigo y metadata liviana de discovery; sin samples/videos locales pesados. |
| `evaluation/` | Codigo, notebooks y comparaciones. |
| `visual_preprocessing/` | Codigo de preprocesamiento visual. |
| `vsr_models/` | Codigo/modelos de VSR, sin checkpoints locales pesados. |
| `segmentacion_oraciones/` | Codigo para cierre/segmentacion futura. |

## Carpetas locales removidas

Estas carpetas ya no existen localmente porque eran datos pesados, caches u outputs
reemplazados por el bucket/manifests finales:

- `data/clips/`
- `data/processed/lip_rois/`
- `dataset/`
- `data/metadata/`
- `data_release/local_sources/`
- `data_release/work/`
- `data_release/cache/`
- `data_release/logs/`
- `data_release/source_metadata/subtitles/`
- `data_discovery/outputs/samples/`
- `data_discovery/outputs/contact_sheets/`
- `data_cleaning_clean_v1/outputs/`
- `data_cleaning_clean_v1/clean_gpt_v1/`
- `data_cleaning_clean_v1/raw_outputs/`
- `data_cleaning_clean_v1/validated/`
- `data_cleaning_clean_v1/manual_gpt_handoff/`
- `data_cleaning_clean_v1/video_jobs/`
- `segmentacion_oraciones/outputs/`

## Docs principales

| Path | Uso |
| --- | --- |
| `README_DATASET.md` | Descripcion del dataset final. |
| `OPEN_ITEMS_DATASET.md` | Pendientes conocidos del dataset. |
| `data_release/reports/HOW_TO_USE_BUCKET.md` | Guia practica para consumir el bucket. |
| `LOCAL_CLEANUP_PLAN.md` | Evidencia del cleanup local ejecutado. |
| `BRANCH_CLEANUP_PLAN.md` | Estado de ramas/tag del cleanup. |
| `REPO_MAP.md` | Este mapa. |

## Manifests finales

| Path | Uso |
| --- | --- |
| `data_release/final_release_manifest.csv` | Manifest general final. |
| `data_release/final_train_manifest_clean_gpt_v1.csv` | Manifest de entrenamiento conservador. |
| `data_release/final_eval_manifest_clean_gpt_v1.csv` | Manifest de evaluacion. |
| `data_release/clean_gpt_manifest.csv` | Estado de limpieza GPT aplicada. |
| `data_release/new_discovery_roi_manifest.csv` | Estado ROI de new discovery. |
| `data_release/asr_large_turbo_manifest.csv` | Estado ASR large/turbo. |

## Desarrollo futuro

El desarrollo futuro de realtime/demo debe ir en una rama nueva y no reabrir el trabajo
de data cleaning local. La fuente canonica de datos grandes es el bucket final.

No relanzar VM, Whisper, ASR, ROI recovery ni GPT manual desde este estado sin una tarea
nueva y explicita.
