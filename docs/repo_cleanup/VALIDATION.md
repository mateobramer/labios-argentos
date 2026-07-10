# VALIDATION — pruebas corridas en `chore/repo-cleanup-safe-v2`

Máquina: MacBook (Darwin 25.5), **sin** los envs `ptt`/`visper`/`mvsr` ni cámara/pesos.
Env disponible: `~/miniforge3/envs/labios` (Python 3.12.13: pandas, numpy, yt_dlp,
whisper; sin cv2/mediapipe/torch/pytest). pytest se instaló **aislado en un dir
temporal** vía `pip --target` (no se modificó el env del usuario).

## Corridas y resultados

| # | Comando | Resultado |
|---|---|---|
| 1 | `git diff --check main...HEAD` | solo whitespace heredado en 6 archivos **porteados byte-idénticos** (preservado a propósito); los archivos nuevos/modificados: limpios |
| 2 | `python3 -m compileall .` (excl. .git/data/dataset) | **OK, exit 0** — todos los `.py` compilan (incl. los 6 de demo modificados y los ~30 porteados) |
| 3 | `pytest data_discovery/tests` | **8 passed** (0.29 s) |
| 4 | `pytest segmentacion_oraciones/tests` | **34 passed** (0.07 s) |
| 5 | `pytest data_cleaning/tests` | 4 errores de colección por `ModuleNotFoundError: cv2` — **idéntico en main** (verificado con checkout a main y re-corrida): pre-existente, NO regresión de esta branch |
| 6 | `descargar_procesar.py --help` | OK: imprime usage |
| 7 | `python -m data_discovery.src.score_candidates --help` | OK: imprime usage (como módulo, convención del repo) |
| 8 | `data_release/scripts/build_release_manifests.py --help` | falla **con mensaje claro e intencional**: `RuntimeError: No se encontro gsutil en PATH` (sin gsutil acá; es el comportamiento documentado "fallar claro sin credenciales") |
| 9 | `data_cleaning_clean_v1/src/validate_patches.py` | espera path posicional (sin --help); comportamiento original de la rama, sin cambios |
| 10 | parseo CSV de todo lo porteado | **23/23 OK** (csv.reader completo) |
| 11 | parseo JSON/JSONL de todo lo porteado | **147/148 OK**; `data_release/reports/vm_run_status.json` tiene BOM UTF-8 (artefacto histórico de VM; parsea con `utf-8-sig`; preservado byte-idéntico) |
| 12 | chequeo de links internos de los 15 `.md` nuevos/tocados | **0 rotos** (los 7 hacia `NEXT_STEPS.md` se resolvieron al crearlo en este mismo commit; re-verificado después) |
| 13 | archivos agregados ≥1 MB vs main | **ninguno** |
| 14 | scan de secretos/credenciales en lo porteado | **sin hallazgos** (los matches de "SECRETOS" son títulos de videos de Telefe) |
| 15 | `git status` al final | limpio (verificado antes del push) |

## Pruebas NO ejecutables acá (con reproducción)

| Prueba | Comando intentado | Error | Dependencia faltante | Impacto | Cómo reproducirla |
|---|---|---|---|---|---|
| Arranque de la demo web | `python demo/demo_web.py --help` | `ModuleNotFoundError: cv2` | env conda `ptt` (OpenCV+MediaPipe) | no se pudo verificar el arranque hasta el punto de cámara | en la máquina de desarrollo: `~/miniconda3/envs/ptt/bin/python demo/demo_web.py` |
| Carga del infer_server | `python demo/infer_server.py` | falta torch/ViSpeR | env `visper` + `~/Desktop/visper` + pesos 1.1 GB | no se verificó CONFIG/READY en vivo | `~/miniconda3/envs/visper/bin/python demo/infer_server.py` (imprime CONFIG y READY sin cámara) |
| Fallback sin Ollama | (requiere lo anterior) | — | ídem | — | con el server arriba y Ollama apagado: `VSR_QWEN=1` debe caer a 1-best sin morir ([SPEC §6](../SPEC.md)) |
| Tests de data_cleaning | `pytest data_cleaning/tests` | `cv2` | ídem `ptt` | 4 módulos de test sin correr (igual que en main) | correr con el env `ptt` |
| score_selftest | — | requiere pesos + clips personales | pesos, datos privados | — | máquina de desarrollo |

**Riesgo residual de los cambios de demo/**: los 6 archivos modificados solo cambian
la *fuente* de 4 constantes de path (env var con default idéntico); compilan y el
default de `REPO` (derivado de `__file__`) es el mismo valor cuando el repo está en
`~/Desktop/labios-argentos`. La verificación en vivo queda pendiente en la máquina
de desarrollo (comando arriba).

## Cómo re-verificar todo rápido

```bash
git fetch origin && git checkout chore/repo-cleanup-safe-v2
git diff main...HEAD --stat | tail -3          # alcance del diff
python3 -m compileall -q . -x "(\.git|data/|dataset/)" && echo OK
python3 -m pytest data_discovery/tests segmentacion_oraciones/tests -q
~/miniconda3/envs/ptt/bin/python demo/demo_web.py    # smoke real (máquina dev)
```
