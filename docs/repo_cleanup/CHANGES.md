# CHANGES — `chore/repo-cleanup-safe-v2` (pasadas 1 y 2)

Base: `origin/main` = `a11f0827` (2026-07-10). **Main no fue modificado nunca.**

## Commits

Pasada 1 (conservadora): `483215b` auditoría/plan · `3b3af25` docs de scope ·
`9990579` porteo de PR #25 · `dff2915` paths por env · `071181c` READMEs ·
`dee9f3d` NEXT_STEPS/CHANGES/VALIDATION.

Pasada 2 (reorganización real): `dd8150c` preprocessing/ · `55e3da2` cleaning/ ·
`8e38ecc` vsr/ · `3eae37f` llm_corrector/+personalization/ · `4f6e673` data_pipeline/
· `1690385` política de datos · `733a5fa` feedback editable · `8040634` envs+setup
· fix resolver VM · docs finales (este commit).

## Árbol raíz: antes → después

```
ANTES (main, 19 entradas):                DESPUÉS (10 carpetas conceptuales):
claude-videos/  curriculum/  data/        cleaning/        (visual_quality + gpt_clean_v1
data_cleaning/  dataset/  demo/  docs/                      + transcript_segmentation)
evaluation/  experiments/                 data/            (solo corpus+manifests+samples)
multilingual-vsr/  new-data-fine-tuning/  data_pipeline/   (sources+descarga+discovery
segmentacion_oraciones/  Survey/                            +release+inventory)
visual_preprocessing/  vsr_models/        demo/            (solo la app integrada)
descargar_procesar.py  README  AGENTS     docs/            (+bibliografia/, repo_cleanup/)
CLAUDE  TO-DO  requirements  .gitignore   experiments/  llm_corrector/
                                          personalization/  preprocessing/  vsr/
                                          + setup.sh  run.sh  envs/  .env.example
```

## Movimientos (todos con `git mv`, ~250 sitios de referencia actualizados)

| Origen | Destino |
|---|---|
| `visual_preprocessing/` | `preprocessing/` |
| `data_cleaning/` | `cleaning/visual_quality/` |
| `data_cleaning_clean_v1/` | `cleaning/gpt_clean_v1/` |
| `segmentacion_oraciones/` | `cleaning/transcript_segmentation/` |
| `vsr_models/{src,splits,runs,README,RUNBOOK}` | `vsr/…` |
| `evaluation/` | `vsr/evaluation/` |
| `curriculum/` | `vsr/curriculum/` |
| `multilingual-vsr/` | `vsr/mpc001/` |
| `new-data-fine-tuning/` | `vsr/historical/ronda2/` |
| `multilingual-vsr/scripts/fase0_llm_correct.py` | `llm_corrector/` |
| `demo/calibracion/` | `personalization/calibracion/` |
| `demo/{build_testset,score_selftest}.py` | `personalization/` |
| `descargar_procesar.py` | `data_pipeline/` |
| `claude-videos/` | `data_pipeline/sources/` |
| `data_discovery/` · `data_release/` · `data_inventory/` | `data_pipeline/{discovery,release,inventory}/` |
| `Survey/` | `docs/bibliografia/` |
| `data/clips/El mensaje de Coscu…` (8 clips) | `data/samples/…` (muestra de smoke) |

Actualizado junto con los moves: imports dotted (`data_cleaning.*`→`cleaning.visual_quality.*`,
`evaluation.*`→`vsr.evaluation.*`, etc.), strings `-m` en subprocess de tests, comandos
en docs/notebooks, `.gitignore`, y el resolver de paths históricos de VM
(`vsr/evaluation/src/batch_vsr_notebook.py`: ancla-vieja→ubicación-nueva, porque los
paths guardados por las VMs son inmutables).

## Datos retirados del árbol de esta rama (política git/bucket)

| Retirado | Tamaño | Recuperación (verificada) |
|---|---|---|
| `data/clips/` (17.078 files) | ~2.24 GB | `git checkout main -- data/clips` · tag · bucket |
| `dataset/` (11.901 files) | ~248 MB | ídem |
| `data/videos/` (9) | ~422 MB | ídem o re-descarga |
| 6 CSVs de análisis en `data/metadata/` | ~54 MB | `git checkout main -- "data/metadata/<f>"` |

El repo pasó de 29.395 a **662 archivos trackeados**. Manifest completo de
recuperación: [`data/README.md`](../../data/README.md). Evidencia de respaldo:
main+tag (bytes exactos, verificado con `git show`), bucket clean-v1 (conteos del
`bucket_validation_report.md` del equipo). La historia de git conserva los blobs
(clonar completo sigue ~9 GB; achicarlo = `filter-repo`, decisión de equipo).

## Duplicados eliminados

- Los 3 módulos de limpieza con nombres distintos quedaron bajo un componente.
- Docs raíz duplicados de la rama #25 (`README_DATASET`, `OPEN_ITEMS`, stub de
  `HOW_TO_USE_BUCKET`, `REPO_MAP` superseded): nunca se portaron (pasada 1).
- No quedaron carpetas README-only ni carpetas vacías.

## Wrappers

Ninguno necesario: todas las referencias se actualizaron en el mismo commit que cada
move. Los únicos "consumidores externos" (VMs históricas) usan paths inmutables que el
resolver mapea (ver arriba).

## Funcionalidad agregada (mínima, pedida)

- **Feedback editable** (`demo/`): ✏️ por segmento → `POST /feedback` → JSONL local
  privado en `data/feedback/` (gitignored; sin código de envío externo).
- `setup.sh`, `run.sh`, `envs/ptt.yml`, `envs/visper.yml`.

## Branches remotas eliminadas (2026-07-10, con verificación por ancestry)

| Rama | SHA final | Por qué era seguro | Recuperación |
|---|---|---|---|
| `chore/reorganizacion` | `27ac2445` | ancestor de main (PR #27) | main |
| `feature/visual-audit-eval-prep` | `2076da55` | ancestor de main (PR #24) | main |
| `fix/llm-correction-review` | `5478c6b5` | ancestor de main | main |
| `realtime/demo-kiosko` | `4385b820` | ancestor de main | main |
| `vsr_models/bigger-finetuning` | `ab81d4b9` | ancestor de main (PR #26) | main |
| `feature/data-discovery-v1` | `93c575fa` | ancestor del tag dataset-clean-v1 | tag |
| `feature/clean-bucket-v1` | `17e44ef6` | ancestor del tag dataset-clean-v1 | tag |
| `feature/full-clean-release` | `7cd9b987` | material liviano portado acá; HEAD congelado en tag nuevo | **tag `archive/full-clean-release`** (creado antes de borrar) + `dataset-clean-v1` |

PR #25: **cerrada como superseded** con explicación. Branches conservadas: `main` y
`chore/repo-cleanup-safe-v2` (este trabajo). Tags: ninguno borrado; +1 creado
(`archive/full-clean-release`).

## Riesgos pendientes / decisión humana

Ver [VALIDATION](VALIDATION.md) §no-ejecutables y [FUTURE_WORK](../FUTURE_WORK.md) §8:
smoke completo de inferencia requiere la máquina con ViSpeR+pesos; reescritura de
historia para achicar el clone; 1 test preexistente en rojo (idéntico en main).
