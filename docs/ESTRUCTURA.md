# Estructura del repo y flujo de datos

Mapa de qué vive dónde y cómo fluye un dato desde YouTube hasta la demo. El modelo
de VSR es ViSpeR y la aproximación a tiempo real es por ventanas con VAD visual.

## Flujo de datos end-to-end

```
1 · SELECCIÓN      data_pipeline/sources/candidatos*.csv  (gate 0: fuentes verificadas a mano)
        │
2 · DESCARGA+CORTE data_pipeline/descargar_procesar.py          yt-dlp → Whisper → ffmpeg
        │            data/videos/<t>/ (no versionado) · data/corpus/<t>/ · data/clips/<t>/ (mp4+txt)
        ▼
3 · PREPROC VISUAL preprocessing/          MediaPipe → warp mean-face → ROI boca 96×96
        │            data/processed/lip_rois/<t>/*.npz  (no versionado, regenerable)
        ▼
4 · CURACIÓN       cleaning/visual_quality/                 detector de clips malos (cara, calidad)
        │            dataset/<t>/  (solo los `keep`)  +  data/metadata/*.csv (manifests)
        ▼
5 · ENTRENAMIENTO  vsr/ (50M Gimeno) · personalization/calibracion/ (LoRA personal ViSpeR)
        │            corre en VMs L4 spot de GCP; pesos a modelos/ (no versionados)
        ▼
6 · EVALUACIÓN     vsr/evaluation/ + personalization/score_selftest.py
        │            WER/CER sobre splits congelados (vsr/splits/) — ledger en docs/RESULTS.md
        ▼
7 · DEMO           demo/demo_web.py               cámara → VAD visual → ViSpeR → [qwen] → subtítulos
```

## Módulos

| Carpeta | Etapa | Contenido |
|---|---|---|
| `data_pipeline/sources/` | 1 | CSVs de fuentes curadas (ronda 1 y 2) con criterios de selección |
| `data_pipeline/descargar_procesar.py` | 2 | único script: descarga, transcribe (Whisper turbo es), corta clips alineados |
| `preprocessing/` | 3 | `src/preprocesar.py`: landmarks → crop 96×96 → `.npz` |
| `cleaning/visual_quality/` | 4 | detección de clips malos, auditorías de calidad visual |
| `cleaning/transcript_segmentation/` | 2b | re-segmentado oracional de transcripciones (sparse, no siempre materializado) |
| `vsr/` | 5 | `src/fine_tune.py` (50M), splits congelados, configs |
| `vsr/curriculum/` | 5b | procesamiento de ViSpeR-es (JSON oficiales → npz) para currículum |
| `vsr/mpc001/` | 5c | notas/scripts sobre la base multilingüe mpc001 (el clon externo no se versiona) |
| `vsr/historical/ronda2/` | 5 (hist.) | corrida completa de la ronda 2 → ft03–ft07; docs de esa fase |
| `vsr/evaluation/` | 6 | evaluación contra test-658, parches al repo de Gimeno |
| `demo/` | 7 | demo web + push-to-talk + streaming, `infer_server.py`, calibración |
| `data_pipeline/discovery/` | 1b | búsqueda/score de fuentes nuevas |
| `data_pipeline/release/` | 4b | manifests + reportes + scripts del release limpio v1 (tag `dataset-clean-v1`; los manifests ≥1 MB solo vía tag/bucket — ver su README) |
| `cleaning/gpt_clean_v1/` | 4b | limpieza GPT de transcripciones del release |
| `data_pipeline/inventory/` | 4b | inventario del bucket del release |
| `data_release/bucket_metadata/` | 4b | espejo local liviano de manifests/reports/metadata de `gs://labios-argentos-vsr-clean-v1` (sin video/audio/ROIs); se actualiza con `scripts/sync_bucket_metadata.py` |
| `scripts/` | — | utilitarios de repo que no pertenecen a un módulo (p. ej. sync de metadata del bucket) |
| `docs/experiments/` | — | **registro de todos los experimentos** con números; empezar por su README |
| `docs/` | — | ARCHITECTURE, SPEC, SETUP, este doc (ESTRUCTURA), RESULTS (**ledger canónico**), METHODOLOGY, LIMITATIONS, PROJECT_EVOLUTION, RESEARCH, SYSTEM_ENGINEERING, DATA_AND_ARTIFACTS, FUTURE_WORK, `experiments/`, `bibliography/`, `archive/` (históricos) |

## Qué se versiona y qué no

**Sí**: código, docs, configs, splits congelados, corpus de transcripciones, manifests
chicos, la muestra de smoke (`data/samples/`, 8 clips) y el espejo liviano de metadata
del bucket en `data_release/bucket_metadata/`.

**No** (ver [`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md)): clips
masivos, videos crudos, ROIs `.npz`, pesos `.pth`, manifests gigantes — viven en los
buckets y quedan recuperables desde el commit pre-limpieza `a11f0827666b11c975df4d8c5b0d6014894e8ee8`/tag (`data/README.md`); clones de repos
externos, venvs, grabaciones personales y modelos calibrados (privacidad).

**Clone liviano**: el árbol actual ya no versiona el dataset masivo, pero la historia
de Git conserva los blobs antiguos y un clone completo puede seguir pesando ~9 GB.
Para una copia liviana usar `git clone --filter=blob:none`; los datos retirados se
recuperan con las instrucciones de [`data/README.md`](../data/README.md).

## Convención para módulos nuevos

```
nombre_modulo/
  README.md     # qué hace y cómo se conecta con el resto
  src/          # lógica reutilizable (import como -m nombre_modulo.src.script)
  notebooks/    # experimentos numerados, si aplica
```

La raíz queda para entrypoints y docs generales. Datos generados grandes: revisar
`.gitignore` y el tamaño del commit antes de agregar. Resultados de experimentos: van a
`docs/experiments/`, no sueltos en el módulo.
