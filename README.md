# labios-argentos

**Lectura de labios (VSR) en español rioplatense, cerca de tiempo real.**

Un sistema completo que mira una boca por webcam —sin audio— y va generando subtítulos:
dataset propio de YouTube rioplatense, fine-tunes de dos familias de modelos, un
corrector LLM local por n-best rescoring, calibración al hablante en ~10 minutos, y una
demo web que corre entera en una laptop M1.

Proyecto académico de Ingeniería en IA (Universidad de San Andrés).

---

## ¿Para qué sirve?

El estado del arte en VSR está casi todo en inglés. En español no existe ningún corpus
audiovisual rioplatense público — este proyecto construye el primero y lo usa para
adaptar y evaluar modelos de lectura de labios. Casos de uso: **accesibilidad** (personas
sin fonación), **entornos ruidosos** donde un micrófono no sirve, y subtitulado en vivo
sin audio.

## Demo

```bash
~/miniconda3/envs/ptt/bin/python demo/demo_web.py            # UI web en http://localhost:8551
~/miniconda3/envs/ptt/bin/python demo/demo_web.py --qwen     # con corrector LLM (también hay toggle en la UI)
~/miniconda3/envs/ptt/bin/python demo/demo_web.py --ckpt modelos/personal/<nombre>.pth   # modelo calibrado
```

Al abrir: 2 segundos de silencio calibran el detector de labios, y después se habla
normal — el sistema corta solo por pausas y va subtitulando. La UI muestra además la
entrada literal del modelo (la tira de recortes de boca 96×96) y el guion acumulado.

**Requisitos:** macOS con webcam (encoder acelerado por MPS en Apple Silicon; en CPU
también corre, más lento), envs conda `ptt` (OpenCV + MediaPipe) y `visper` (PyTorch +
ESPnet), el repo ViSpeR en `~/Desktop/visper` con sus pesos `visper_vsr_base.pth`
(1.1 GB, no se versionan), y opcionalmente [Ollama](https://ollama.com) con
`qwen3:4b-instruct-2507-q4_K_M` para el corrector.

## Arquitectura

```
cámara (30 fps) ─────────────────────────────▶ MJPEG a la UI
   │
   ▼
MediaPipe FaceLandmarker (hasta 3 caras; sticky-lock a la más cercana)
   │  apertura de labios
   ▼
VAD visual (auto-calibrado; corta por pausa 0.45 s / tope 4 s)
   │  segmento de habla
   ▼
crop de boca 96×96 (warp mean-face) → .npz
   │
   ▼
infer_server (env visper — ViSpeR 288M, conformer)
   ├─ encoder ──── MPS   (~0.17 s)
   └─ beam search ─ CPU  (beam 3, ~0.9 s)
   │
   ▼
[opcional] qwen3:4b n-best rescoring (Ollama local, +1.2 s, −3.0 WER)
   │
   ▼
UI web (SSE): subtítulo sobre el video + tira de boca + guion acumulado
```

**Latencia total: ~1.1 s por segmento (~2.3 s con el corrector)** en una MacBook M1.
Cada decisión de esta configuración (beam 3, encoder en MPS, top-5 con el 4b, VAD por
pausas) sale de un experimento medido — el detalle está en [`docs/SPEC.md`](docs/SPEC.md).

## Resultados

Sobre `test-658` (658 clips de YouTube, 2 hablantes held-out, speaker-independent) y
sobre el self-test (100 clips propios grabados en condiciones controladas):

| Modelo | %WER test-658 | %WER self-test |
|---|---|---|
| ft05 — mejor fine-tune propio (50M, LIP-RTVE + dataset AR ~19 h) | 65.05 | ~68 |
| **ViSpeR zero-shot** (288M, 794 h de pre-entrenamiento en español) | **45.22** | 29.51 |
| ViSpeR + qwen n-best rescoring | — | **26.46** (−3.04, significativo) |
| ViSpeR + LoRA personal (60 clips del hablante) | 44.54 | 24.51 en test personal |

Las dos conclusiones grandes: **la escala de pre-entrenamiento domina** (794 h zero-shot
le gana por ~20 WER a nuestro mejor fine-tune con 19 h), y **la corrección LLM solo
funciona como n-best rescoring** (la corrección 1-best siempre empeora; el rescoring da
−3.04 WER con IC95 pareado que excluye 0). Tabla maestra completa e índice de los 10
experimentos: [`experiments/README.md`](experiments/README.md). Ledger vivo:
[`docs/RESULTS.md`](docs/RESULTS.md).

## Calibración al hablante

Desde la propia UI (`/calibrar`): la persona graba ~40 frases push-to-talk en el
browser, y `bash demo/calibracion/calibrar_entrenar.sh <nombre>` entrena un LoRA en una
VM L4 spot de GCP (~10 min, ~$0.05, se autodestruye) y descarga el modelo personal.
Validado en [`experiments/10`](experiments/10_adaptacion_hablante.md): −4.7 WER personal
sin olvidar el test general. La misma página tiene el modo "Ayudanos a entrenar" para
donar pares clip+texto al dataset (quedan locales, no se versionan).

## Estructura del repo

| Carpeta | Qué es |
|---|---|
| `demo/` | Demo web + push-to-talk + streaming, servidor de inferencia, calibración |
| `experiments/` | Registro completo de experimentos con resultados (empezar por su README) |
| `docs/` | [`SPEC.md`](docs/SPEC.md) (especificación), [`ESTRUCTURA.md`](docs/ESTRUCTURA.md) (mapa y flujo de datos), [`RESULTS.md`](docs/RESULTS.md) (ledger) |
| `descargar_procesar.py` | Etapa 1 del pipeline de datos: YouTube → clips alineados con texto |
| `visual_preprocessing/` | Etapa 2: clips → ROIs de boca 96×96 (`.npz`) |
| `data_cleaning/` | Etapa 3: detección y descarte de clips malos |
| `claude-videos/` | Selección curada de fuentes (gate 0 del dataset) |
| `vsr_models/` | Fine-tuning del modelo 50M (Gimeno) + splits congelados |
| `evaluation/` | Evaluación WER/CER contra test-658 |
| `curriculum/` | Procesamiento de datos ViSpeR-es para currículum de pre-entrenamiento |
| `multilingual-vsr/` | Notas y scripts sobre la base multilingüe (el clon del repo externo no se versiona) |
| `new-data-fine-tuning/` | Corrida histórica de la ronda 2 de datos (ft03–ft07) |
| `data/` | Dataset generado (clips + corpus versionados; videos crudos y `.npz` no) |

El flujo de datos completo y las convenciones para agregar código están en
[`docs/ESTRUCTURA.md`](docs/ESTRUCTURA.md). La guía para agentes/colaboradores, en
[`AGENTS.md`](AGENTS.md).

## Costos

La inferencia es 100 % local ($0, sin datos a terceros). Los entrenamientos corrieron en
VMs L4 spot de GCP (g2-standard-8, ~$0.28/h spot): cada fine-tune del 50M costó del orden
de $1–3, y una calibración personal ~$0.05. Todo el proyecto se hizo dentro de créditos
educativos/promocionales.

## Limitaciones (honestas)

- El modelo es **offline/bidireccional**: la demo aproxima tiempo real cortando por
  pausas de labios, no es streaming causal cuadro a cuadro.
- WER ~26–30 en condiciones ideales (buena luz, boca frontal, habla clara); ~45 en
  YouTube variado. La lectura de labios pura sigue siendo un problema difícil.
- Validado en profundidad con un solo hablante para la calibración personal.
- El encoder acelerado requiere Apple Silicon (MPS); en otras plataformas corre en CPU.
