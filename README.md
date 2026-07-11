# labios-argentos

**Reconocimiento visual del habla (VSR) en español rioplatense: evaluación de modelos VSR,
rescoring con LLM, adaptación por hablante y una demo local cercana a tiempo real.**

Un sistema que mira una boca por webcam —sin audio— y genera subtítulos: dataset propio de
YouTube rioplatense, evaluación de dos familias de modelos, un corrector LLM local por
n-best rescoring, calibración al hablante en ~10 minutos, y una demo web que corre entera
en una laptop.

<!-- Demo en video/GIF: pendiente de grabar (ver docs/FUTURE_WORK.md). -->

`VSR` · `español rioplatense` · `ViSpeR` · `LLM n-best rescoring` · `LoRA por hablante` · `demo local`

---

## 1. Qué es

Reconocimiento visual del habla (lip reading) para español rioplatense: construir el
dataset, evaluar qué modelos funcionan en el acento, estudiar cuándo un LLM ayuda a
corregir la salida, adaptar el modelo a un hablante, e integrar todo en una demo local que
subtitula en vivo.

## 2. Motivación

El estado del arte en VSR está casi todo en inglés y no identificamos un corpus audiovisual
público específicamente orientado al español rioplatense. Este proyecto construye uno y lo
usa para adaptar y evaluar modelos. Casos de uso: **accesibilidad** (personas sin fonación),
**entornos ruidosos** donde un micrófono no sirve, y subtitulado en vivo sin audio.

## 3. Hallazgos principales

- **La escala de pre-entrenamiento domina.** ViSpeR (288M, 794 h en español) zero-shot le
  gana por ~20 WER al mejor fine-tune propio (50M, ~19 h de datos).
- **La corrección LLM sobre 1-best empeoró el WER en todas las condiciones evaluadas.** El
  **n-best rescoring** produjo una mejora significativa en el régimen de CER bajo evaluado:
  −3.04 WER con IC95 pareado que excluye 0 (n=100).
- **La personalización con LoRA mostró una mejora de 4.7 puntos de WER en el hablante
  evaluado**, sin degradación del test general; la mejora personal todavía no fue
  estadísticamente significativa con la muestra actual. El full-fine-tuning degradó
  severamente el modelo de 288M.

Todos los números y su significancia: [`docs/RESULTS.md`](docs/RESULTS.md) (ledger
canónico). Cómo se miden: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## 4. Arquitectura

```
cámara → MediaPipe FaceLandmarker → VAD visual (corta por pausas) →
crop de boca 96×96 → ViSpeR 288M (encoder MPS + beam CPU) →
[opcional] qwen3:4b n-best rescoring → subtítulos en UI web
```

Latencia total **~1.1 s por segmento (~2.3 s con el corrector)** en una MacBook M1. Cada
decisión de configuración sale de un experimento medido. Detalle:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) y [`docs/SPEC.md`](docs/SPEC.md).

## 5. Demo

```bash
bash run.sh                # UI web en http://localhost:8551
bash run.sh --qwen         # con corrector LLM (también hay toggle en la UI)
```

Al abrir: 2 s de silencio calibran el detector de labios; después se habla normal y el
sistema corta solo por pausas y va subtitulando. La UI muestra la entrada literal del
modelo (la tira de recortes de boca 96×96), el guion acumulado, y permite **corregir cada
predicción** (queda local). Instalación y requisitos: [`docs/SETUP.md`](docs/SETUP.md).

## 6. Resultados

Sobre `test-658` (658 clips, 2 hablantes held-out, speaker-independent) y el self-test
(100 clips propios en condiciones controladas):

| Modelo | %WER test-658 | %WER self-test |
|---|---|---|
| Mejor fine-tune propio (50M, LIP-RTVE + ~19 h AR) | 65.05 | ~68 |
| **ViSpeR zero-shot** (288M, 794 h español) | **45.22** | 29.51 |
| ViSpeR + n-best rescoring (qwen) | — | **26.46** (−3.04, significativo) |
| ViSpeR + LoRA personal (60 clips del hablante) | 44.54 | 24.51 (test personal) |

Índice de experimentos: [`docs/experiments/`](docs/experiments/). Ledger:
[`docs/RESULTS.md`](docs/RESULTS.md).

## 7. Quickstart

```bash
git clone <repo> && cd labios-argentos
bash setup.sh              # crea los envs conda (ptt, visper) y verifica artefactos
bash run.sh                # levanta la demo
```

Requiere macOS con webcam; el encoder se acelera en Apple Silicon (MPS) y también corre en
CPU (más lento). Pesos de ViSpeR y detalles: [`docs/SETUP.md`](docs/SETUP.md).

## 8. Estructura del repo

El pipeline, en orden:

| # | Carpeta | Componente |
|---|---|---|
| 1 | `data_pipeline/` | fuentes, descarga+corte, discovery, release, inventario |
| 2 | `cleaning/` | control de calidad: limpieza visual, GPT de transcripciones, segmentación |
| 3 | `preprocessing/` | landmarks → warp mean-face → ROI de boca 96×96 |
| 4 | `vsr/` | modelos VSR, fine-tuning, splits congelados, evaluación WER/CER |
| 5 | `llm_corrector/` | n-best rescoring con LLM (y los resultados negativos) |
| 6 | `personalization/` | calibración por hablante (grabación, LoRA, evaluación) |
| 7 | `demo/` | app integrada en vivo + feedback editable |

Además: `data/` (artefactos livianos + muestra), `docs/`, `envs/`. Los directorios se
numeran en la documentación para contar el flujo; en disco conservan nombres importables.

## 9. Datos y modelos

El repositorio versiona **solo artefactos livianos**: transcripciones, manifests chicos y
una muestra mínima para smoke tests. El **dataset completo, los ROIs y los pesos viven
fuera de Git** (buckets del proyecto + tag `dataset-clean-v1`). Qué está incluido, qué no,
y cómo recuperarlo: [`data/README.md`](data/README.md) y
[`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md).

## 10. Reproducibilidad

Entornos reproducibles en [`envs/`](envs/) (`ptt.yml`, `visper.yml`); `setup.sh` los crea y
`run.sh` levanta la demo. La inferencia es 100 % local ($0, sin datos a terceros). Los
entrenamientos corrieron en VMs L4 spot de GCP (~$1–3 por fine-tune, ~$0.05 por calibración
personal). Protocolo de evaluación (splits congelados, normalización, bootstrap):
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## 11. Citation

Si usás este trabajo, citá según [`CITATION.cff`](CITATION.cff).

## 12. Autores

Desarrollado por Martín Bianchi, Federico Gutman, Joaquín Szterensus, Santiago Bunge y
Mateo Bramer (Universidad de San Andrés). Detalles y forma de citar:
[`CITATION.cff`](CITATION.cff).
