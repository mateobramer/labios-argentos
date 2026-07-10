# DATA_AND_ARTIFACTS — fuentes de verdad de datos, pesos y artefactos

El repo versiona **solo una parte** de los datos. Este doc dice qué vive dónde, cómo
obtener lo que falta, y qué hacer cuando un artefacto no está.

## Tipos de artefacto y su fuente de verdad

| Artefacto | Qué es | Fuente de verdad | ¿Versionado en git? |
|---|---|---|---|
| **Clips alineados** | mp4 + txt por fuente (`data/clips/`, `dataset/`) | este repo (main) | ✅ (~2.9 GB — ver "política" abajo) |
| **Corpus / transcripciones** | cache de Whisper por fuente (`data/corpus/`) | este repo | ✅ |
| **Manifests** | CSVs de estado del dataset (`data/metadata/`) | este repo | ✅ |
| **Splits congelados** | `vsr/splits/` — test-658/val, fijos desde ft03 | este repo | ✅ **no tocar** |
| **Videos crudos** | fuente completa de YouTube (`data/videos/`) | YouTube (regenerables con `descargar_procesar.py`) | ❌ gitignored |
| **ROIs `.npz`** | crops de boca 96×96 | regenerables con `preprocessing/` | ❌ gitignored |
| **Pesos base ViSpeR** | `visper_vsr_base.pth` (1.1 GB) | repo ViSpeR/TII + copia en bucket | ❌ |
| **Pesos fine-tuneados** | ft03–ft07, LoRAs | `gs://labios-argentos-vsr-dataset` | ❌ **irreemplazables** |
| **Modelos personales** | LoRA por hablante (`modelos/personal/`) | máquina local de cada persona | ❌ privacidad |
| **Grabaciones personales** | `~/vsr_personal/`, `~/vsr_contrib/` | máquina local | ❌ privacidad, **nunca** |
| **Release limpio v1** | dataset re-construido + limpiado (jul 2026) | bucket + tag `dataset-clean-v1` | parcial (ver abajo) |

## Los tres buckets (GCP, proyecto `visual-speech-recognition-nlp`)

| Bucket | Propósito |
|---|---|
| `gs://labios-argentos-vsr-dataset` | canónico de entrenamiento: pesos, dataset para VMs (AGENTS.md) |
| `gs://labios-argentos-vsr-data` | datos de trabajo de la fase full-clean-release |
| `gs://labios-argentos-vsr-clean-v1` | release limpio v1 |

No renombrar ni migrar sin decisión explícita. Acceso: cuenta del proyecto GCP
(los scripts asumen `gcloud`/`gsutil` autenticados; sin credenciales fallan con el
error de auth de gsutil — no hay fallback, es intencional).

## El release limpio v1 y el tag `dataset-clean-v1`

La rama `feature/full-clean-release` (PR #25) construyó un release limpio del dataset
(re-construcción de fuentes + ASR + limpieza GPT + discovery de fuentes nuevas). Sus
piezas livianas (reportes, scripts, manifests <1 MB) están portadas acá en
`data_release/`, `data_discovery/`, `cleaning/gpt_clean_v1/`, `data_inventory/`.

- **Manifests grandes** (10 CSVs, ~82 MB — `final_release_manifest.csv`, etc.):
  NO están en esta rama. Recuperarlos: `git show dataset-clean-v1:data_release/<nombre>.csv`
  (el tag los preserva) o desde el bucket clean-v1.
- **Datos pesados del release**: solo en bucket (`HOW_TO_USE_BUCKET.md` portado en
  `data_release/reports/`).

## Política de versionado — y la decisión abierta

**Main hoy**: los clips mp4+txt SÍ se versionan ("son el dataset", `docs/ESTRUCTURA.md`);
crudos/ROIs/pesos no. Consecuencia: clone completo ~9 GB → **usar sparse-checkout**
(ya configurado en los clones del equipo; `git sparse-checkout add <dir>` si un path
trackeado no aparece).

**La rama full-clean-release propone lo contrario**: datos pesados solo en bucket,
`.gitignore` bloqueando `*.mp4`. Esta limpieza **no** adoptó ese cambio (habría
des-oficializado el dataset versionado sin decisión del equipo). Queda como decisión
abierta en [`NEXT_STEPS.md`](NEXT_STEPS.md) §7 con los trade-offs.

## Configuración y variables de entorno

Paths y knobs de la demo (defaults = comportamiento actual; ver `.env.example`):

| Variable | Default | Qué controla |
|---|---|---|
| `LABIOS_REPO` | raíz del repo (derivada del script) | paths de datos/modelos de la demo |
| `VISPER_DIR` | `~/Desktop/visper` | clon del repo ViSpeR con sus pesos |
| `VISPER_PY` | `~/miniconda3/envs/visper/bin/python` | intérprete del env de inferencia |
| `VSR_CKPT` | (vacío = pesos base) | state_dict alternativo (modelo personal) |
| `VSR_BEAM` / `VSR_QWEN` / `VSR_QMODEL` / `VSR_MPS` | 3 / off / qwen3:4b-instruct… / auto | knobs de inferencia ([SPEC §3](SPEC.md)) |

Los scripts de GCP (`personalization/calibracion/*.sh`, `vsr/historical/ronda2/scripts/*.sh`) son
templates de startup de VM con sus propios paths internos de VM — no usan estas variables.

## Cómo montar un entorno desde cero (resumen)

1. Clonar con sparse-checkout (o aceptar los ~9 GB).
2. Envs conda: `ptt` (OpenCV+MediaPipe) y `visper` (PyTorch+ESPnet) — [README](../README.md) §Requisitos.
3. Clonar ViSpeR en `~/Desktop/visper` (o `VISPER_DIR`) y bajar `visper_vsr_base.pth`
   (1.1 GB; del repo ViSpeR/TII o del bucket del proyecto).
4. Opcional: Ollama + `ollama pull qwen3:4b-instruct-2507-q4_K_M`.
5. `python demo/demo_web.py` — si falta un artefacto, el error dice cuál
   (sin mocks: la ausencia de datos reales no se disimula).

## Privacidad

- Grabaciones personales y modelos calibrados: **nunca** al repo ni a buckets públicos.
- El dataset proviene de YouTube público; los manifests citan fuente por video.
- La inferencia de la demo no manda nada afuera (127.0.0.1, [SPEC §7](SPEC.md)).
