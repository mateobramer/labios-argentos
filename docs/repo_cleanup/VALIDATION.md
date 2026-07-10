# VALIDATION — pruebas corridas en `chore/repo-cleanup-safe-v2` (pasada 2)

Máquina: MacBook M5 (Darwin 25.5). Envs usados: **`ptt` creado en esta pasada**
(`~/miniforge3/envs/ptt`: py3.11, cv2 5.0.0, mediapipe 0.10.35, torch, pandas, pyyaml,
matplotlib, pytest — reproducible con `envs/ptt.yml`) y `labios` preexistente (torch,
whisper; no se modificó). No hay en esta máquina: clon ViSpeR + pesos, gsutil/gcloud,
Ollama, permiso de cámara para procesos background.

## Corridas y resultados (pasada 2)

| # | Comando | Resultado |
|---|---|---|
| 1 | `pytest cleaning data_pipeline/discovery/tests` (env ptt) | **110 passed, 1 failed** — la falla (`test_builder_no_modifica_splits_canonicos`) es **preexistente: falla idéntico en main** (verificado con checkout a main + re-corrida, 1 failed/68 passed allí). Las 5 regresiones que introdujo la reorg se detectaron con esta suite y se corrigieron (gitignore batch_vsr, resolver de VM, strings `-m`) |
| 2 | `python3 -m compileall .` | OK exit 0 (todo el árbol) |
| 3 | **Smoke preprocessing end-to-end REAL** sobre `data/samples/`: `procesar_clip()` | **OK: 137 ROIs 96×96, 99 % detección de cara** — valida muestra + mediapipe + cadena de imports post-reorg |
| 4 | `demo/demo_web.py --help` (env ptt) | OK: usage completo |
| 5 | Arranque demo (`--no-open --port 8599`) con `VISPER_PY` override | **llegó hasta la cámara** ("OpenCV: not authorized to capture video… requesting") — el override de env var funciona; bloqueado por permiso de cámara de macOS para procesos background + inferencia real |
| 6 | `infer_server.py` con `VISPER_DIR` inexistente | falla clara y esperada: `ModuleNotFoundError: datamodule` (falta el clon ViSpeR; sin mocks) |
| 7 | `descargar_procesar.py --help` · `-m data_pipeline.discovery.src.score_candidates --help` | OK ambos |
| 8 | Recuperación del tag: `git show dataset-clean-v1:data_release/final_release_manifest.csv` | **OK — devuelve el CSV** (comando del manifest de recuperación verificado en vivo) |
| 9 | Ancestry de las 8 ramas antes de borrar (`git merge-base --is-ancestor`) | 5 ancestor de main, 2 ancestor del tag, full-clean-release congelada en `archive/full-clean-release` |
| 10 | `git check-ignore` de la muestra | `data/samples/*.mp4` NO ignorada (whitelist funciona) |
| 11 | Links internos de todos los `.md` tocados | 0 rotos (re-chequeado tras cada fase) |
| 12 | Archivos agregados ≥1 MB vs main | ninguno |
| 13 | Scan de secretos en todo lo agregado/movido | sin hallazgos |
| 14 | `git status` final | limpio antes del push |
| 15 | `git diff --check main...HEAD` | solo whitespace heredado en archivos porteados byte-idénticos (a propósito) |

## No ejecutables en esta máquina (bloqueados, con reproducción exacta)

| Prueba | Bloqueada por | Cómo reproducirla (máquina con ViSpeR) |
|---|---|---|
| infer_server hasta `CONFIG`/`READY` | falta clon ViSpeR (`~/Desktop/visper`, incluye `visper_zeroshot.py` propio) + `visper_vsr_base.pth` 1.1 GB (bucket `gs://labios-argentos-vsr-dataset` o release TII) + env `visper` (spec: `envs/visper.yml`) | `~/miniconda3/envs/visper/bin/python demo/infer_server.py` → debe imprimir CONFIG y READY |
| Fallback sin Ollama | requiere lo anterior | con server arriba y Ollama apagado, `VSR_QWEN=1`: cae a 1-best sin morir (SPEC §6) |
| Con/sin corrector (WER en vivo) | ídem + Ollama + cámara | `bash run.sh` vs `bash run.sh --qwen` |
| Cámara en vivo | permiso de macOS a procesos background | correr `bash run.sh` desde Terminal con permiso de cámara |
| Scripts de release contra bucket | sin gsutil/credenciales acá | falla intencional clara: "No se encontro gsutil en PATH" (verificado) |

## Cómo re-verificar rápido

```bash
git fetch && git checkout chore/repo-cleanup-safe-v2
bash setup.sh                                     # crea envs ptt/visper si faltan
~/miniforge3/envs/ptt/bin/python -m pytest cleaning data_pipeline/discovery/tests -q
python3 -m compileall -q . && echo OK
bash run.sh                                       # demo real (con cámara y ViSpeR)
```

---

## Pasada 3 — validación (2026-07-10, posicionamiento público)

| # | Comando | Resultado |
|---|---|---|
| 1 | links markdown de todo el repo | **0 rotos** |
| 2 | `python3 -m compileall .` | **OK exit 0** |
| 3 | `pytest cleaning data_pipeline/discovery/tests` (env ptt) | **110 passed, 1 failed** — la falla (`test_builder_no_modifica_splits_canonicos`) es **preexistente e idéntica en main** |
| 4 | `yaml.safe_load(CITATION.cff)` | OK, 3 autores, cff 1.2.0 |
| 5 | smoke preprocessing sobre `data/samples/` | OK: 137 ROIs 96×96, 99 % detección |
| 6 | scan de términos internos (Claude/Fable/scratchpad/TP/rúbrica) en docs públicos | **0** (excluyendo `experiments/`+`archive/` históricos, que se reencuadraron pero preservan evidencia) |
| 7 | scan de nombres de carpeta viejos en código/docs | 0 (fuera de los docs de recuperación, que deben mencionarlos) |
| 8 | archivos ≥1 MB **agregados** en esta pasada | **ninguno** (los >1 MB — notebooks, `face_landmarker.task`, CSVs de batch_vsr — son preexistentes) |
| 9 | scan de secretos (`AIza…`, AWS, PRIVATE KEY) | 1 match investigado y **descartado**: es data base64 de imagen embebida en un output `text/html` de notebook, no una credencial (verificado: "AIza" aparece en medio de un blob base64, no es una key de 39 chars) |
| 10 | `git diff --check` | limpio |

**Smoke de demo/infer_server hasta CONFIG/READY**: sigue bloqueado en esta máquina (falta
el clon ViSpeR + pesos) — reproducción en la sección anterior. Sin cambios de pasada 3 que
lo afecten (la demo compila y `--help` funciona).
