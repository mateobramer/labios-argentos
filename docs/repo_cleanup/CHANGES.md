# CHANGES — qué cambió exactamente en `chore/repo-cleanup-safe-v2`

Base: `origin/main` = `a11f0827` (2026-07-10). Nada en main fue modificado.
Commits (en orden):

1. `483215b` — docs: auditoría y plan de limpieza (repo_cleanup/)
2. `3b3af25` — docs: scope del proyecto y guías de los dos TPs
3. `9990579` — port: material único de feature/full-clean-release (sin datos pesados)
4. `dff2915` — demo: paths configurables por env con defaults actuales + .env.example
5. `071181c` — docs: READMEs de módulos faltantes y enlaces cruzados
6. (este commit) — docs: NEXT_STEPS, CHANGES y VALIDATION

## Estructura antes → después

**Sin movimientos**: la estructura de main (reorganizada en PR #27, mergeada hoy mismo)
se conservó intacta. Cambios = **solo agregados** + 8 archivos modificados:

```
(nuevo) .env.example
(nuevo) data_pipeline/discovery/            53 archivos — portado completo
(nuevo) data_pipeline/release/             ~121 archivos — portado parcial (<1 MB c/u)
(nuevo) cleaning/gpt_clean_v1/    14 archivos — portado parcial
(nuevo) data_pipeline/inventory/             3 archivos — portado completo
(nuevo) demo/README.md, vsr/curriculum/README.md, vsr/mpc001/README.md,
        vsr/historical/ronda2/README.md, docs/bibliografia/README.md
(nuevo) docs/{PROJECT_SCOPE,RESEARCH_TP,ENGINEERING_TP,DATA_AND_ARTIFACTS,NEXT_STEPS}.md
(nuevo) docs/repo_cleanup/{AUDIT,MIGRATION_PLAN,CHANGES,VALIDATION}.md
(mod)   README.md, docs/ESTRUCTURA.md    — enlaces a lo nuevo, ledger declarado canónico
(mod)   requirements.txt                 — + faster-whisper (dep de data_pipeline/discovery)
(mod)   demo/{demo_web,demo_ptt,demo_stream,build_testset,infer_server,score_selftest}.py
        — paths por env var con defaults idénticos (ver abajo)
(mod)   data_pipeline/release/README.md, cleaning/gpt_clean_v1/README.md
        — nota de contexto sobre manifests grandes excluidos (única alteración
          respecto de la rama de origen; el resto es byte-idéntico)
```

## Lista de movimientos

Ninguno (`git mv`: no se usó — no hizo falta mover nada).

## Archivos archivados

Ninguno nuevo (lo histórico ya estaba señalizado: `docs/archivo/`,
`vsr/historical/ronda2/` — este último ahora tiene README que lo marca como histórico).

## Archivos eliminados

**Ninguno.** Política aplicada: nada en main calificó como inequívocamente inútil.

## Wrappers agregados

Ninguno (no se movió ningún entrypoint; no hicieron falta).

## Paths modificados (demo/, defaults sin cambio de comportamiento)

| Archivo | Antes | Después |
|---|---|---|
| `demo_web.py`, `demo_ptt.py`, `demo_stream.py`, `build_testset.py` | `REPO = ~/Desktop/labios-argentos` fijo | `LABIOS_REPO` env, default = raíz del repo derivada de `__file__` |
| ídem + `demo_web/ptt/stream` | `VISPER_PY = ~/miniconda3/envs/visper/bin/python` fijo | `VISPER_PY` env, mismo default |
| `infer_server.py` | `REPO = ~/Desktop/visper` fijo | `VISPER_DIR` env, mismo default |
| `score_selftest.py` | 3 paths fijos | `MVSR_REPO`, `VISPER_DIR`, `LABIOS_REPO`/`FT05_CKPT` env, mismos defaults |

Sin cambios en: parámetros de VAD/beam/corrector/calibración, splits, normalización,
scripts `.sh` de GCP, algoritmos.

## Branches de las que se portó material

Solo `feature/full-clean-release` (PR #25; incluye todo lo de
`feature/data-discovery-v1` y `feature/clean-bucket-v1`, que están 100 % contenidas).
Las otras 5 ramas remotas están 100 % mergeadas en main (ver [AUDIT §5](AUDIT.md)).

## Material NO portado y motivo (recuperación indicada)

| Qué | Motivo | Cómo recuperarlo |
|---|---|---|
| Borrados masivos de `data/` y `dataset/` (~29.000 archivos) | decisión de datos separada; reproducirla acá cambiaría la política de main sin decisión del equipo | n/a (es una eliminación; main los conserva) |
| `.gitignore` de la rama (bloquea `*.mp4` etc.) | contradice la política vigente de main (clips versionados) | `git show origin/feature/full-clean-release:.gitignore` |
| 12 manifests CSV ≥1 MB de `data_pipeline/release/` (~127 MB) | peso; el árbol de trabajo no los necesita | `git show dataset-clean-v1:data_pipeline/release/<nombre>.csv` |
| `cleaning/gpt_clean_v1/patch_log.csv` (5.2 MB) | peso | `git show dataset-clean-v1:cleaning/gpt_clean_v1/patch_log.csv` |
| Raíz: `README_DATASET.md`, `OPEN_ITEMS_DATASET.md` | byte-idénticos a `data_pipeline/release/reports/` (portados ahí) | ya están en `data_pipeline/release/reports/` |
| Raíz: `HOW_TO_USE_BUCKET.md` | stub que apunta a la versión canónica de `reports/` (portada) | `git show origin/feature/full-clean-release:HOW_TO_USE_BUCKET.md` |
| `REPO_MAP.md` | describe la estructura pre-reorganización; superseded por `docs/ESTRUCTURA.md` | `git show origin/feature/full-clean-release:REPO_MAP.md` |
| `BRANCH_CLEANUP_PLAN.md`, `LOCAL_CLEANUP_PLAN.md` | planes de trabajo internos de esa rama, no del estado actual | `git show origin/feature/full-clean-release:<nombre>.md` |

**Importante**: no se borró ninguna rama ni tag. Todo lo no-portado sigue existiendo
en `origin/feature/full-clean-release` y en el tag `dataset-clean-v1`.

## Riesgos pendientes

- La contradicción de política de datos main↔PR#25 sigue abierta (documentada en
  [DATA_AND_ARTIFACTS](../DATA_AND_ARTIFACTS.md) y [NEXT_STEPS §7](../NEXT_STEPS.md)).
- Los docs portados de la rama describen flujos que asumen el bucket clean-v1 y
  `gsutil` autenticado — correrán solo con credenciales del proyecto GCP.
- `demo/` no pudo validarse end-to-end en la máquina de esta limpieza (no tiene los
  envs `ptt`/`visper` ni cámara/pesos) — ver [VALIDATION](VALIDATION.md) §no-ejecutables.
- Whitespace heredado en 6 archivos porteados (trailing space / blank line EOF) — se
  preservaron byte-idénticos a la rama de origen a propósito.
