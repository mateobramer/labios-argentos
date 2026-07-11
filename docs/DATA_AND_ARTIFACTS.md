# DATA_AND_ARTIFACTS — fuentes de verdad de datos, pesos y artefactos

El repo versiona **solo una parte** de los datos. Este doc dice qué vive dónde, cómo
obtener lo que falta, y qué hacer cuando un artefacto no está.

## Tipos de artefacto y su fuente de verdad

| Artefacto | Qué es | Fuente de verdad | ¿Versionado en git? |
|---|---|---|---|
| **Clips alineados** | mp4 + txt por fuente | bucket clean-v1 (release) + commit pre-limpieza `a11f0827666b11c975df4d8c5b0d6014894e8ee8` + tag `dataset-clean-v1` (bytes históricos exactos) | ❌ en el árbol actual (muestra de 8 clips en `data/samples/` — ver [`data/README.md`](../data/README.md)) |
| **Corpus / transcripciones** | cache de Whisper por fuente (`data/corpus/`) | este repo | ✅ (texto) |
| **Manifests chicos** | `data/metadata/` (<2 MB c/u) | este repo | ✅ |
| **Manifests gigantes** | análisis de calidad (~54 MB) y release (~127 MB) | bucket + tag `dataset-clean-v1` | ❌ (retirados; recuperación en [`data/README.md`](../data/README.md)) |
| **Splits congelados** | `vsr/splits/` — test-658/val, fijos desde ft03 | este repo | ✅ **no tocar** |
| **Videos crudos** | fuente completa de YouTube (`data/videos/`) | YouTube (regenerables con `data_pipeline/descargar_procesar.py`) | ❌ gitignored |
| **ROIs `.npz`** | crops de boca 96×96 | regenerables con `preprocessing/` | ❌ gitignored |
| **Pesos base ViSpeR** | `visper_vsr_base.pth` (1.1 GB) | repo ViSpeR/TII + copia en bucket | ❌ |
| **Pesos fine-tuneados** | ft03–ft07, LoRAs | `gs://labios-argentos-vsr-dataset` | ❌ **irreemplazables** |
| **Modelos personales** | LoRA por hablante (`modelos/personal/`) | máquina local de cada persona | ❌ privacidad |
| **Grabaciones personales** | `~/vsr_personal/`, `~/vsr_contrib/` | máquina local | ❌ privacidad, **nunca** |
| **Release limpio v1** | dataset re-construido y limpiado, empaquetado como release | bucket + tag `dataset-clean-v1` | parcial (ver abajo) |

## Los tres buckets (GCP, proyecto `visual-speech-recognition-nlp`)

| Bucket | Rol | Contiene |
|---|---|---|
| `gs://labios-argentos-vsr-dataset` | **fuente canónica de entrenamiento** | pesos ft03–ft07 y LoRAs (**irreemplazables**), dataset empaquetado para VMs |
| `gs://labios-argentos-vsr-data` | **workspace histórico** de la fase full-clean-release | intermedios de trabajo (regenerables desde fuentes + scripts) |
| `gs://labios-argentos-vsr-clean-v1` | **release limpio v1** (congelado, tag `dataset-clean-v1`) | clips mp4 (12.112 existing + 13.193 new_discovery), ROIs npz, transcripts, `manifests/`, `reports/` — conteos validados en [`bucket_validation_report.md`](../data_pipeline/release/reports/bucket_validation_report.md) |

No renombrar ni migrar sin decisión explícita. Acceso: cuenta del proyecto GCP
(los scripts asumen `gcloud`/`gsutil` autenticados; sin credenciales fallan con el
error de auth de gsutil — no hay fallback, es intencional).

## El release limpio v1 y el tag `dataset-clean-v1`

El release limpio v1 (re-construcción de fuentes + ASR + limpieza GPT + discovery de
fuentes nuevas) vive parcialmente en este repo: sus piezas livianas (reportes,
scripts, manifests <1 MB) están en `data_pipeline/release/`, `data_pipeline/discovery/`,
`cleaning/gpt_clean_v1/`, `data_pipeline/inventory/`.

- **Manifests grandes** (10 CSVs, ~82 MB — `final_release_manifest.csv`, etc.):
  NO están en el árbol actual. Recuperarlos: `git show dataset-clean-v1:data_release/<nombre>.csv`
  (el tag los preserva) o desde el bucket clean-v1.
- **Datos pesados del release**: solo en bucket (`HOW_TO_USE_BUCKET.md` en
  `data_pipeline/release/reports/`).

## Política de versionado

**Git**: código, documentación, configs, splits congelados, manifests chicos,
muestra mínima de smoke (`data/samples/`, 2.9 MB) y reportes.
**Bucket**: videos, clips masivos, ROIs, checkpoints y manifests gigantes.
**Solo local/privado**: grabaciones personales, modelos calibrados, feedback de la demo.

`data/clips/`, `dataset/`, `data/videos/` y los manifests grandes de calidad visual
no están en el árbol principal. Nada se borró de la historia de Git, del tag ni de
los buckets — el manifest de recuperación está en [`data/README.md`](../data/README.md).
El `.gitignore` bloquea media/pesos nuevos y whitelistea `data/samples/`.

**Nota**: los clips del bucket clean-v1 son la forma *release* (curada, re-construida)
y no un espejo 1:1 de `data/clips` en su versión original — para los bytes históricos
exactos del árbol original, la fuente es Git (commit pre-limpieza `a11f0827666b11c975df4d8c5b0d6014894e8ee8`
o tag `dataset-clean-v1`).

La historia de git conserva los blobs pesados: clonar todo el repo sigue pesando
~9 GB. Achicarlo de verdad exige reescritura de historia (`git filter-repo`),
pendiente de decisión de equipo. Mientras tanto, `git clone --filter=blob:none` da
un clone liviano y el árbol actual ya no materializa datos masivos en el working tree.

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

1. Clonar liviano: `git clone --filter=blob:none https://github.com/mateobramer/labios-argentos.git`.
2. Crear los envs con `bash setup.sh`; detalles en [`SETUP.md`](SETUP.md).
3. Clonar ViSpeR en `~/Desktop/visper` (o `VISPER_DIR`) y bajar `visper_vsr_base.pth`
   (1.1 GB; del repo ViSpeR/TII o del bucket del proyecto).
4. Opcional: Ollama + `ollama pull qwen3:4b-instruct-2507-q4_K_M`.
5. `python demo/demo_web.py` — si falta un artefacto, el error dice cuál
   (sin mocks: la ausencia de datos reales no se disimula).

## Privacidad

- Grabaciones personales y modelos calibrados: **nunca** al repo ni a buckets públicos.
- El dataset proviene de YouTube público; los manifests citan fuente por video.
- La inferencia de la demo no manda nada afuera (127.0.0.1, [SPEC §7](SPEC.md)).
