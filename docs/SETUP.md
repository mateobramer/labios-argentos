# SETUP — entornos, pesos y comandos

Guía de instalación y ejecución. Para el mapa del repo ver [`ESTRUCTURA.md`](ESTRUCTURA.md);
para qué datos viven dónde, [`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md).

## Instalación rápida

```bash
bash setup.sh     # crea los envs conda (ptt, visper) desde envs/*.yml y verifica artefactos
bash run.sh       # levanta la demo web (http://localhost:8551)
```

`setup.sh` es idempotente: no toca envs que ya existan.

## Entornos

| Entorno | Para qué | Spec |
|---|---|---|
| `ptt` | demo web y captura (OpenCV + MediaPipe) | [`envs/ptt.yml`](../envs/ptt.yml) |
| `visper` | inferencia ViSpeR (PyTorch + ESPnet) | [`envs/visper.yml`](../envs/visper.yml) |
| `mvsr` | espnet1 vendoreado (mpc001) — corre el modelo 50M/ft05 local vía remap | reconstruible desde el repo mpc001 |
| entrenamiento | solo en VMs de GCP: fine-tuning del 50M | ver `04_vsr/` (nombre lógico) → `vsr/` |

## Pesos y artefactos externos (no se versionan)

- **`visper_vsr_base.pth`** (1.1 GB): pesos base de ViSpeR. Van dentro del clon del repo
  ViSpeR en `~/Desktop/visper` (o `$VISPER_DIR`). Origen: release de ViSpeR (TII) o la
  copia del proyecto en `gs://labios-argentos-vsr-dataset`.
- **`face_landmarker.task`** (MediaPipe): viene versionado en `preprocessing/models/`;
  `setup.sh` lo baja si falta.
- **Corrector LLM (opcional)**: [Ollama](https://ollama.com) +
  `ollama pull qwen3:4b-instruct-2507-q4_K_M`.

## Configuración (variables de entorno)

Todos los paths y knobs tienen default = comportamiento estándar. Ver
[`.env.example`](../.env.example) para la lista completa (`LABIOS_REPO`, `VISPER_DIR`,
`VISPER_PY`, `VSR_BEAM`, `VSR_QWEN`, `VSR_QMODEL`, `VSR_CKPT`, `VSR_MPS`, …).

## Comandos frecuentes

```bash
# Demo web (--qwen activa el corrector; --ckpt carga un modelo personal)
bash run.sh                     # equivale a  <env ptt>/python demo/demo_web.py
bash run.sh --qwen

# Pipeline de datos para una fuente nueva
python data_pipeline/descargar_procesar.py "URL_YOUTUBE"
python -m preprocessing.src.preprocesar "<titulo>"
python -m cleaning.visual_quality.src.detectar_clips_malos "<titulo>" [--materializar]

# Calibración al hablante (después de grabar en la UI /calibrar)
bash personalization/calibracion/calibrar_entrenar.sh <nombre>

# Scoring del self-test
<env visper>/python personalization/score_selftest.py --model visper

# Tests
<env ptt>/python -m pytest cleaning data_pipeline/discovery/tests -q
```

> Nota: los directorios se numeran 01→07 en la documentación para contar el flujo del
> pipeline, pero en disco conservan nombres importables por Python (`data_pipeline/`,
> `cleaning/`, …). Ver [`ARCHITECTURE.md`](ARCHITECTURE.md).
