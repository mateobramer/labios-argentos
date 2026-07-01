# realtime

Servicio de **lectura de labios en vivo** (demo kiosko): te ponés frente a la cámara,
mantenés apretado un botón mientras hablás, y te transcribe. Es el primer eslabón del
sistema en tiempo real que define `ESTRUCTURA_PROYECTO.md`.

## Qué hace (v1 — demo)

```
navegador (webcam, push-to-talk)
   └─ graba un clip  ──POST /transcribe──▶  FastAPI
                                              ├─ preprocess_live: cuadros → ROI labial (T,96,96) @25fps
                                              ├─ infer: ROI → texto crudo (ft03 + beam search ESPnet)
                                              └─ corrector: texto crudo → texto corregido (Claude, opcional)
   ◀──────────  {texto, texto_crudo, métricas}  ──┘
```

### Decisiones de diseño (ver discusión en el proyecto)

- **Por clips, no causal.** El modelo (`Conv3D-ResNet18 → Conformer → CTC/attention`) es
  **bidireccional**. En vez de hacerlo causal (cuesta accuracy y datos), corre sobre la
  utterance completa: lookahead acotado al clip, **cero cirugía al modelo**, y la entrada
  queda en la misma distribución que su entrenamiento (clips de 3–10 s).
- **Push-to-talk.** Sin audio no hay forma trivial de segmentar dónde empieza/termina una
  frase. Mantener apretado resuelve el borde sin VAD. El **VAD visual** (apertura bucal con
  los landmarks que ya extraemos) queda para v2.
- **Reuso total.** El preproc reusa `visual_preprocessing` (mismo warp a cara media) y la
  inferencia reusa el `Speech2Text` de Gimeno (mismo decode que el zero-shot/eval). El ROI
  que produce en vivo es **idéntico** al del dataset offline (validado: misma hipótesis).
- **Corrector LLM** (`corrector.py`), opcional. Un LLM reconstruye la oración rioplatense
  más probable a partir de la hipótesis cruda. **Es una capa de legibilidad, no de
  exactitud**: medido sobre el test rioplatense, el corrector NO baja el WER (prompted
  empeora +1–2 pts; el techo teórico rescoreando el N-best es −6 pts) — el cuello es el
  VSR. **Graceful**: si el backend no está disponible, queda OFF y la demo sigue andando
  con el texto crudo. La UI muestra el antes/después (crudo tachado + corregido).

## Estructura

```
realtime/
  src/
    preprocess_live.py   # clip → ROI (T,96,96) @25fps   [reusa visual_preprocessing]
    infer.py             # ROI → texto crudo              [Speech2Text de Gimeno + ft03]
    corrector.py         # texto crudo → corregido        [Claude, opcional/graceful]
    server.py            # FastAPI: GET / (kiosko) + POST /transcribe
  web/index.html         # UI kiosko (webcam + push-to-talk + transcripción)
  models/
    vsr_config.yaml      # config del modelo (token_list char embebido)
    ft03_best.pth        # checkpoint fine-tuneado (gitignored, *.pth)
  requirements.txt
```

## Correr la demo

```bash
# 1) env (una vez) — ver requirements.txt para el detalle
conda create -y -n realtime python=3.11 && conda activate realtime
pip install -r realtime/requirements.txt

# 2) modelo de Gimeno (una vez): provee ASRTask + Speech2Text
git clone --depth 1 https://github.com/david-gimeno/evaluating-end2end-spanish-lipreading.git \
  ~/evaluating-end2end-spanish-lipreading

# 3) el checkpoint fine-tuneado va en realtime/models/ft03_best.pth
#    (gitignored; bajarlo de gs://labios-argentos-vsr-data/models/ o copiarlo a mano)

# 4) (opcional) corrector LLM. Default: Ollama local (basta el server corriendo).
#    Para usar Claude en su lugar:
export CORRECTOR_BACKEND=claude
export ANTHROPIC_API_KEY=sk-ant-...
export CORRECTOR_MODEL=claude-haiku-4-5   # recomendado para el kiosko (latencia)

# 5) levantar el servicio (desde la raíz del repo)
python -m realtime.src.server
# abrir http://localhost:8000
```

### Configuración (variables de entorno)

| var | default | qué es |
|---|---|---|
| `GIMENO_REPO` | `~/evaluating-end2end-spanish-lipreading` | repo del modelo (ASRTask/Speech2Text) |
| `VSR_CKPT` | `realtime/models/ft03_best.pth` | checkpoint a cargar |
| `VSR_DEVICE` | `cpu` | `cpu` anda < tiempo real en el Mac; `mps`/`cuda` opcional |
| `VSR_BEAM` | `10` | beam size del decode (↑ = mejor/lento) |
| `CORRECTOR_BACKEND` | `ollama` | backend del corrector: `ollama` (local, gratis) o `claude` (API) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | modelo local (backend ollama) |
| `ANTHROPIC_API_KEY` | — | credencial (solo backend claude; si falta, corrector OFF) |
| `CORRECTOR_MODEL` | `claude-opus-4-8` | modelo Claude (kiosko: `claude-haiku-4-5` por latencia) |
| `CORRECTOR_ENABLED` | `1` | `0` desactiva el corrector por completo |
| `PORT` | `8000` | puerto del server |

## Rendimiento (Mac M2, CPU, beam 10)

Modelo carga en ~1.8 s. Por clip de ~4 s: preproc ~1.2 s + inferencia ~1.5 s → **RTF ≈ 0.7**
(más rápido que tiempo real, **sin** corrector). El corrector LLM agrega la latencia de una
llamada a la API (por eso conviene `claude-haiku-4-5` en el kiosko).

## Próximos pasos (hacia el paper)

- **Mejorar el VSR base** — es el cuello confirmado: swap de backbone (MOSEAS / auto-AVSR),
  re-labels con Whisper large-v3, adaptación al hablante del kiosko.
- **VAD visual** para sacar el push-to-talk (apertura bucal con los landmarks de mediapipe).
- **Student causal** destilado del modelo bidireccional → streaming real de baja latencia.
- ~~RL / few-shot del corrector~~ — investigado y **descartado como lever de WER**: el
  corrector entrenado empeoró +14 pts y el techo del N-best es −6. Queda solo como capa
  cosmética opcional de la demo.
