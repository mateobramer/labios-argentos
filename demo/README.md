# demo/ — sistema integrado de lectura de labios en vivo

El entregable de ingeniería: cámara → VAD visual → crop de boca → ViSpeR → [qwen] →
subtítulos. Doc técnico: [`docs/SPEC.md`](../docs/SPEC.md); guía del TP:
[`docs/ENGINEERING_TP.md`](../docs/ENGINEERING_TP.md).

| Archivo | Qué es | Env |
|---|---|---|
| `demo_web.py` | **la demo principal**: UI web en `http://localhost:8551` (MJPEG + SSE, solo stdlib) | `ptt` |
| `infer_server.py` | proceso hijo con ViSpeR cargado; protocolo stdin/stdout ([SPEC §3.1](../docs/SPEC.md)) | `visper` (lo spawnea la demo) |
| `demo_ptt.py` | demo push-to-talk por teclado (ventana cv2) — precursor de la web | `ptt` |
| `demo_stream.py` | demo streaming por VAD en ventana cv2 — precursor de la web | `ptt` |
| `build_testset.py` | graba el self-test leyendo frases (append + resumible) | `ptt` |
| `score_selftest.py` | evalúa WER/CER del self-test con ViSpeR o ft05 | `visper` / `mvsr` |
| `calibracion/` | splits + orquestador del LoRA personal en GCP ([exp. 10](../experiments/10_adaptacion_hablante.md)) | — |
| `web/` | HTML/JS de la UI | — |

```bash
~/miniconda3/envs/ptt/bin/python demo/demo_web.py           # demo
~/miniconda3/envs/ptt/bin/python demo/demo_web.py --qwen    # con corrector LLM
```

Paths configurables por env (`LABIOS_REPO`, `VISPER_DIR`, `VISPER_PY`, …): ver
[`.env.example`](../.env.example). Requisitos completos: [README raíz](../README.md).
