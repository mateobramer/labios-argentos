# MIGRATION_PLAN — plan de la limpieza (branch `chore/repo-cleanup-safe-v2`)

Base: `origin/main` = `a11f0827`. Ver diagnóstico completo en [`AUDIT.md`](AUDIT.md).

Principio rector: main fue reorganizado hoy (PR #27) y su estructura ya es coherente.
**No se re-mueven módulos existentes.** Esta branch: (1) porta el material único de
`feature/full-clean-release`, (2) cierra huecos de documentación para los dos TPs,
(3) desacopla paths hardcodeados con defaults idénticos, (4) valida todo.

## Estructura destino

Idéntica a main, más:

```
data_discovery/            ← portado de feature/full-clean-release (completo, 0.8 MB)
data_release/              ← portado PARCIAL (<1 MB por archivo; manifests grandes via tag)
cleaning/gpt_clean_v1/    ← portado parcial (sin patch_log.csv 5.3 MB)
data_inventory/            ← portado completo (11 KB)
docs/
  PROJECT_SCOPE.md         ← nuevo
  RESEARCH_TP.md           ← nuevo (evidencia LLM×VSR)
  ENGINEERING_TP.md        ← nuevo (sistema/demo)
  DATA_AND_ARTIFACTS.md    ← nuevo (fuentes de verdad de datos)
  NEXT_STEPS.md            ← nuevo
  repo_cleanup/            ← AUDIT, MIGRATION_PLAN, CHANGES, VALIDATION
demo/README.md             ← nuevo
curriculum/README.md       ← nuevo
multilingual-vsr/README.md ← nuevo
new-data-fine-tuning/README.md ← nuevo
Survey/README.md           ← nuevo
.env.example               ← nuevo (paths/knobs de la demo)
```

## Movimientos exactos

Ninguno dentro de main (sin `git mv` de módulos existentes — la estructura ya es la
correcta y moverla rompería los comandos documentados en README/SPEC/experiments).

## Porteos desde `feature/full-clean-release` (commit del tag `dataset-clean-v1` = 7221b55 y HEAD 7cd9b98)

| Origen (rama #25) | Destino | Criterio |
|---|---|---|
| `data_discovery/**` | ídem | completo — pipeline con src+tests, 0.8 MB |
| `data_release/**` con blob <1 MB | ídem | reports, scripts, README, manifests chicos |
| `data_release/*.csv` ≥1 MB (10 archivos, ~82 MB) | **NO** | recuperables: `git show dataset-clean-v1:<path>`; se listan en `data_release/README` porteado y en CHANGES |
| `cleaning/gpt_clean_v1/**` salvo `patch_log.csv` (5.3 MB) | ídem | src, prompts, reports, rejected_patches.jsonl (evidencia negativa) |
| `data_inventory/**` | ídem | completo |
| raíz: `README_DATASET.md`, `OPEN_ITEMS_DATASET.md` | **NO** (duplicados byte-idénticos de `data_release/reports/`) | se porta solo la copia de reports/ |
| raíz: `HOW_TO_USE_BUCKET.md` | comparar con `data_release/reports/HOW_TO_USE_BUCKET.md`, portar la versión más completa | difieren |
| raíz: `REPO_MAP.md`, `BRANCH_CLEANUP_PLAN.md`, `LOCAL_CLEANUP_PLAN.md` | **NO** — describen la estructura pre-reorg / el workflow de esa rama; superseded | recuperables desde la rama; anotado en CHANGES |
| `requirements.txt`: `faster-whisper>=1.2.1` | se agrega (dependencia de data_discovery) | |
| `.gitignore` de la rama | **NO** — contradice la política de datos de main (decisión abierta, va a NEXT_STEPS) | |
| Borrados masivos de `data/` y `dataset/` | **NO** — decisión de datos separada, fuera de scope | |

## Consolidaciones

- Fuente única de resultados: `docs/RESULTS.md` se declara **ledger canónico** (ya lo
  dice AGENTS.md); los docs nuevos linkean, no copian tablas.
- `docs/RESEARCH_TP.md` indexa la evidencia LLM×VSR ya existente
  (`experiments/04`, `09 §LLM`, `10 §redundancia`, notebooks 03/08 de cleaning/visual_quality).
- `docs/ENGINEERING_TP.md` indexa SPEC + experiments 06/09/10 + demo/.

## Eliminaciones

Ninguna. Nada en main califica como "inequívocamente inútil" (la reorg de hoy ya
limpió). Los no-porteos de la rama #25 NO son eliminaciones (la rama y el tag quedan
intactos en remoto).

## Wrappers de compatibilidad

No se necesitan: no se mueve ningún entrypoint existente.

## Cambios de paths (demo)

Patrón: extender el mecanismo ya existente (`VSR_*` env vars) a los paths, con
**defaults idénticos** al comportamiento actual:

| Archivo | Constante | Cambio |
|---|---|---|
| `demo/demo_web.py`, `demo_ptt.py`, `demo_stream.py`, `build_testset.py` | `REPO` | `LABIOS_REPO` env → default: raíz del repo derivada de `__file__` (equivale al valor actual cuando el repo está en `~/Desktop/labios-argentos`, y corrige el caso de clones en otra ubicación) |
| ídem | `VISPER_PY` | `VISPER_PY` env → default actual sin cambios |
| `demo/infer_server.py` | `REPO` | `VISPER_DIR` env → default `~/Desktop/visper` sin cambios |
| `demo/score_selftest.py` | paths de repos/ckpt | env overrides análogos, defaults sin cambios |
| `.env.example` | — | documenta todas las variables (paths + `VSR_QWEN/BEAM/QMODEL`) |

Sin cambios en: parámetros de VAD/beam/corrector/calibración, scripts `.sh` de GCP
(templates de VM, documentados en DATA_AND_ARTIFACTS), splits, normalización.

## Validaciones previstas

1. `git diff --check` (whitespace).
2. `python3 -m compileall` de todo el árbol.
3. Tests existentes: `cleaning/visual_quality/tests/`, `cleaning/transcript_segmentation/tests/`,
   `data_discovery/tests/` (porteados) — con pytest si el env lo permite.
4. `--help`/arranque de entrypoints hasta donde no exijan cámara/pesos/envs conda.
5. Chequeo de links internos de todos los `.md` nuevos y tocados.
6. Grep de referencias a paths viejos/movidos (no debería haber: no se mueve nada).
7. Parseo de manifests chicos porteados (csv.reader sobre cada uno).
8. Verificación de que ningún archivo agregado supere 1 MB.
9. `git status` limpio al final.

## Riesgos y rollback

- Riesgo principal: los porteos de la rama #25 introducen docs que mencionan la política
  "datos al bucket" que contradice main → mitigado con nota de contexto en
  `data_release/README.md` porteado y la decisión abierta en NEXT_STEPS.
- Riesgo menor: derivar `REPO` de `__file__` cambia el path efectivo si alguien ejecuta
  el script desde una copia suelta fuera del repo → mitigado con `LABIOS_REPO` env.
- Rollback: la branch no toca main; borrar la branch = rollback total. Cada fase es un
  commit separado y revertible individualmente.

## Orden de commits

1. `docs: auditoría y plan de limpieza (repo_cleanup/)` ← este commit
2. `docs: scope del proyecto y guías de los dos TPs`
3. `port: material único de feature/full-clean-release (sin datos pesados)`
4. `demo: paths configurables por env con defaults actuales + .env.example`
5. `docs: READMEs de módulos faltantes y enlaces cruzados`
6. `docs: NEXT_STEPS, CHANGES y VALIDATION`
