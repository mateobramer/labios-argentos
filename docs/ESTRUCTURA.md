# Estructura del repo y flujo de datos

Mapa de qué vive dónde y cómo fluye un dato desde YouTube hasta la demo. Reemplaza a los
viejos `FLUJO.md`, `ESTRUCTURA_PROYECTO.md` y `PIPELINE_PROYECTO.md` (borrados en la
reorganización de 2026-07; el plan original Auto-AVSR quedó obsoleto — hoy el mejor
modelo es ViSpeR y la aproximación a tiempo real es por ventanas con VAD visual).

## Flujo de datos end-to-end

```
1 · SELECCIÓN      claude-videos/candidatos*.csv  (gate 0: fuentes verificadas a mano)
        │
2 · DESCARGA+CORTE descargar_procesar.py          yt-dlp → Whisper → ffmpeg
        │            data/videos/<t>/ (no versionado) · data/corpus/<t>/ · data/clips/<t>/ (mp4+txt)
        ▼
3 · PREPROC VISUAL visual_preprocessing/          MediaPipe → warp mean-face → ROI boca 96×96
        │            data/processed/lip_rois/<t>/*.npz  (no versionado, regenerable)
        ▼
4 · CURACIÓN       data_cleaning/                 detector de clips malos (cara, calidad)
        │            dataset/<t>/  (solo los `keep`)  +  data/metadata/*.csv (manifests)
        ▼
5 · ENTRENAMIENTO  vsr_models/ (50M Gimeno) · demo/calibracion/ (LoRA personal ViSpeR)
        │            corre en VMs L4 spot de GCP; pesos a modelos/ (no versionados)
        ▼
6 · EVALUACIÓN     evaluation/ + demo/score_selftest.py
        │            WER/CER sobre splits congelados (vsr_models/splits/) — ledger en docs/RESULTS.md
        ▼
7 · DEMO           demo/demo_web.py               cámara → VAD visual → ViSpeR → [qwen] → subtítulos
```

## Módulos

| Carpeta | Etapa | Contenido |
|---|---|---|
| `claude-videos/` | 1 | CSVs de fuentes curadas (ronda 1 y 2) con criterios de selección |
| `descargar_procesar.py` | 2 | único script: descarga, transcribe (Whisper turbo es), corta clips alineados |
| `visual_preprocessing/` | 3 | `src/preprocesar.py`: landmarks → crop 96×96 → `.npz` |
| `data_cleaning/` | 4 | detección de clips malos, auditorías de calidad visual |
| `segmentacion_oraciones/` | 2b | re-segmentado oracional de transcripciones (sparse, no siempre materializado) |
| `vsr_models/` | 5 | `src/fine_tune.py` (50M), splits congelados, configs |
| `curriculum/` | 5b | procesamiento de ViSpeR-es (JSON oficiales → npz) para currículum |
| `multilingual-vsr/` | 5c | notas/scripts sobre la base multilingüe mpc001 (el clon externo no se versiona) |
| `new-data-fine-tuning/` | 5 (hist.) | corrida completa de la ronda 2 → ft03–ft07; docs de esa fase |
| `evaluation/` | 6 | evaluación contra test-658, parches al repo de Gimeno |
| `demo/` | 7 | demo web + push-to-talk + streaming, `infer_server.py`, calibración |
| `data_discovery/` | 1b | búsqueda/score de fuentes nuevas (portado de `feature/full-clean-release`) |
| `data_release/` | 4b | manifests + reportes + scripts del release limpio v1 (tag `dataset-clean-v1`; los manifests ≥1 MB solo vía tag/bucket — ver su README) |
| `data_cleaning_clean_v1/` | 4b | limpieza GPT de transcripciones del release (portado, ídem) |
| `data_inventory/` | 4b | inventario del bucket del release |
| `experiments/` | — | **registro de todos los experimentos** con números; empezar por su README |
| `docs/` | — | SPEC, este doc, RESULTS (**ledger canónico**), PROJECT_SCOPE, RESEARCH_TP, ENGINEERING_TP, DATA_AND_ARTIFACTS, NEXT_STEPS, `archivo/` (históricos), `repo_cleanup/` (auditoría 2026-07) |

## Qué se versiona y qué no

**Sí**: código, clips alineados (`data/clips/`, mp4+txt — son el dataset), corpus/
transcripciones (cache), manifests y splits, documentación y experimentos.

**No** (`.gitignore`): videos crudos (`data/videos/`, regenerables con yt-dlp), ROIs
`.npz` y pesos `.pth` (regenerables/pesados; los pesos viven en el bucket
`gs://labios-argentos-vsr-dataset`), clones de repos externos, venvs, grabaciones
personales y modelos calibrados (privacidad).

**Sparse-checkout**: el repo trackea ~29k archivos (dataset incluido); el working copy
solo materializa las carpetas en `git sparse-checkout list`. Si un path trackeado no
aparece en disco, agregá su carpeta con `git sparse-checkout add <dir>`.

## Convención para módulos nuevos

```
nombre_modulo/
  README.md     # qué hace y cómo se conecta con el resto
  src/          # lógica reutilizable (import como -m nombre_modulo.src.script)
  notebooks/    # experimentos numerados, si aplica
```

La raíz queda para entrypoints y docs generales. Datos generados grandes: revisar
`.gitignore` y el tamaño del commit antes de agregar. Resultados de experimentos: van a
`experiments/`, no sueltos en el módulo.
