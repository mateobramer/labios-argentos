# personalization/ — calibración y adaptación al hablante

Componente de adaptación personal del modelo (LoRA sobre ViSpeR). Evidencia:
[`../docs/experiments/10`](../docs/experiments/10_adaptacion_hablante.md) — −4.7 WER personal sin
olvido del test general; el full-FT colapsa el 288M (por eso LoRA r16/α32).

| Archivo | Qué hace | Env |
|---|---|---|
| `build_testset.py` | graba frases push-to-talk (append+resumible) → npz + manifest local | `ptt` |
| `score_selftest.py` | evalúa WER/CER del set personal con ViSpeR o ft05 | `visper` / `mvsr` |
| `calibracion/armar_splits_cal.py` | arma splits personales desde las tomas | — |
| `calibracion/calibrar_entrenar.sh` | orquesta el LoRA en VM L4 spot de GCP (~10 min, ~$0.05, se autodestruye) y baja el modelo | local + GCP |
| `calibracion/startup_cal_template.sh` | startup script de la VM (template, paths internos de VM) | GCP |

## Flujo

1. En la demo web, `/calibrar`: la persona graba ~40 frases (la UI usa los mismos
   PROMPTS de `build_testset.py`).
2. `bash personalization/calibracion/calibrar_entrenar.sh <nombre>`.
3. La demo carga el resultado con `--ckpt modelos/personal/<nombre>.pth` (o `VSR_CKPT`).

La demo **invoca** este componente; la ciencia y los scripts viven acá.

## Privacidad (regla dura)

Grabaciones (`~/vsr_personal/`, `~/vsr_contrib/`) y modelos calibrados
(`modelos/personal/`) **no se versionan nunca**. Para entrenar suben al bucket privado
del proyecto y el modelo vuelve a la máquina local. Ver [CONTRIBUTING](../CONTRIBUTING.md).
