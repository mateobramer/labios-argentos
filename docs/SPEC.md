# SPEC — especificación del sistema de demo

Especificación técnica de la demo web de lectura de labios (`demo/`). El principio de
diseño del sistema: **ninguna decisión es "porque sí"** — cada parámetro sale de un
experimento medido (referencias a `experiments/` en cada sección).

## 1. Alcance

Subtitulado cerca de tiempo real por lectura de labios (sin audio), local en una laptop
M1, con corrector LLM opcional y calibración al hablante. El modelo (ViSpeR 288M) es
offline/bidireccional: el "tiempo real" se aproxima cortando el habla en segmentos por
pausas de labios e infiriendo cada segmento completo (~1.1 s después del corte).

## 2. Componentes

```
browser ◀── MJPEG /video · SSE /events ──▶ demo_web.py (env ptt)          infer_server.py (env visper)
                                            ├─ hilo_camara    (captura)     ├─ ViSpeR 288M conformer
                                            ├─ hilo_landmarks (MediaPipe)   ├─ encoder → MPS
                                            ├─ hilo_segmentador (VAD+corte) ├─ beam search → CPU
                                            └─ HTTP (stdlib Threading)      └─ qwen n-best (Ollama, opcional)
                                                     │  stdin/stdout (pipe)      ▲
                                                     └───────────────────────────┘
```

- **`demo/demo_web.py`** (env `ptt`): captura, landmarks, VAD visual, crop de boca
  96×96, servidor HTTP (solo stdlib, `ThreadingHTTPServer` en `127.0.0.1:8551`).
- **`demo/infer_server.py`** (env `visper`): proceso hijo con el modelo cargado en
  memoria; recibe paths `.npz` por stdin y devuelve texto por stdout.
- **Ollama** (opcional): `qwen3:4b-instruct-2507-q4_K_M` para el n-best rescoring.
- **`personalization/calibracion/`**: splits + orquestador de entrenamiento LoRA en GCP.

## 3. Interfaces

### 3.1 Protocolo del infer_server (stdin/stdout, una línea por mensaje)

| Dirección | Mensaje | Significado |
|---|---|---|
| server → cliente | `CONFIG {json}` | antes de READY: `{encoder, beam, qwen, modelo}` (clientes viejos la ignoran) |
| server → cliente | `READY` | modelo cargado, listo para inferir |
| cliente → server | `<path.npz>` | inferir ese clip; responde una línea con el texto |
| cliente → server | `::qwen 0\|1` | toggle del corrector en runtime; responde `::ok qwen=N` |

Variables de entorno: `VSR_BEAM` (default 3; 5 si `VSR_QWEN=1`), `VSR_QWEN`,
`VSR_QMODEL`, `VSR_CKPT` (state_dict alternativo, p. ej. modelo personal), `VSR_MPS`
(=0 fuerza CPU). La web fija beam 5 siempre para poder togglear qwen sin reiniciar.

### 3.2 Endpoints HTTP de la demo

| Método | Path | Función |
|---|---|---|
| GET | `/` · `/calibrar` | UI principal · UI de calibración |
| GET | `/video` | stream MJPEG de la cámara (con recuadro de lock si hay >1 cara) |
| GET | `/events` | SSE con el estado (fase, movimiento de labios, segmentos, config, n_caras) |
| GET | `/strip/<id>` | tira JPEG de 7 ROIs del segmento `<id>` ("lo que ve el modelo") |
| POST | `/qwen` | toggle del corrector (proxy del `::qwen` al server) |
| POST | `/clear` | limpia el guion acumulado |
| GET/POST | `/cal/estado` · `/cal/entrar` · `/cal/rec` · `/cal/corte` · `/cal/salir` | flujo de calibración/contribución |

## 4. Flujo y latencias medidas (MacBook M1, mediana con higiene de CPU)

| Etapa | Latencia | Dónde se midió |
|---|---|---|
| VAD visual: cierre de segmento tras la pausa | 0.45 s (parámetro) | [06](../experiments/06_demo_y_remap.md) |
| Encoder conformer (288M) en **MPS** | **0.17 s** (0.58 s en CPU) | [09](../experiments/09_velocidad_inferencia.md) |
| Beam search (beam 3, CPU) | ~0.9 s | [09](../experiments/09_velocidad_inferencia.md) |
| **Total por segmento (sin LLM)** | **~1.1 s** | [09](../experiments/09_velocidad_inferencia.md) |
| qwen n-best rescoring (Ollama caliente) | +1.24 s (frío: 3.7 s) | [09](../experiments/09_velocidad_inferencia.md) |
| **Total con corrector** | **~2.3 s** | [09](../experiments/09_velocidad_inferencia.md) |

## 5. Decisiones de diseño y su justificación experimental

| Decisión | Alternativas descartadas | Evidencia |
|---|---|---|
| beam = 3 (2.2× más rápido, mismo WER) | beam 40 (igual WER, 3×), beam 1-2 (peor) | [09](../experiments/09_velocidad_inferencia.md) sweep completo con IC |
| encoder en MPS, beam en CPU | todo en MPS (espnet rompe por device mismatch); int8 (Pareto-dominado en M1); CTC-greedy (+12 WER) | [09](../experiments/09_velocidad_inferencia.md) |
| corrector = **n-best rescoring** top-5 con qwen3:4b | corrección 1-best (siempre empeora); top-10, scores en el prompt, qwen 9b (no mejoran) | [04](../experiments/04_qwen_corrector.md) — −3.04 WER significativo (IC95 pareado [+0.71, +5.53], n=100) |
| VAD **visual** por apertura de labios, auto-calibrado (2 s de silencio; umbral = max(sens×ruido, piso); pausa 0.45 s; tope 4 s) | VAD por audio (no hay audio); ventana fija (corta palabras) | [06](../experiments/06_demo_y_remap.md) |
| modelo base = ViSpeR 288M zero-shot | ft05 propio (65 vs 45 WER; y 2.3× más lento por su LM externo); full-FT de ViSpeR (overfitea) | [02](../experiments/02_zeroshot.md), [03](../experiments/03_visper_finetunes.md), [09](../experiments/09_velocidad_inferencia.md) |
| calibración = **LoRA** (r16/α32, lr 1e-4, augment) en L4 spot (~10 min, ~$0.05) | full-FT lr1e-5 (receta del 50M: **colapsa** el 288M); entrenar local sin GPU (horas) | [10](../experiments/10_adaptacion_hablante.md) |
| multi-cara: sticky-lock a la cara más grande, re-lock a 1.5 s | tomar siempre la primera detección (salta entre personas del público) | diseño para demo en vivo, [06](../experiments/06_demo_y_remap.md) |

## 6. Robustez y manejo de errores

- **Ollama apagado / error del LLM** → fallback silencioso al 1-best del beam (la demo
  nunca se cae por el corrector).
- **Beam en CPU siempre**: mover el decoder a MPS rompe espnet (device mismatch) — está
  fijado por diseño, no por accidente.
- **Warmup de cámara**: el primer `cap.read()` de macOS suele fallar; se tolera y se
  descartan frames iniciales.
- **Warmup de MPS**: los primeros clips compilan kernels por forma → los primeros
  segmentos son más lentos; es esperado.
- **Calibración**: valida ≥20 frames y ≥60 % de detección de cara antes de guardar una
  toma; el VAD y la poda de segmentos quedan desactivados durante la calibración.
- **>1 cara en cámara**: recuadro "leyendo a esta persona" + aviso en la UI; el crop no
  salta de hablante.

Pendientes conocidos (ver [TO-DO.md](../TO-DO.md)): reinicio automático si muere el
infer_server, mensajes guiados para cámara sin permiso, tests automatizados.

## 7. Privacidad y costos

Todo corre local: ni video ni texto salen de la máquina ($0 marginal por inferencia).
Las grabaciones de calibración quedan en `~/vsr_personal/` (fuera del repo); para
entrenar suben al bucket privado del proyecto y el modelo resultante vuelve a
`modelos/personal/` (gitignored). Entrenamiento de calibración: ~$0.05 por persona en
una VM spot que se autodestruye.

## 8. Limitaciones

- No es streaming causal: es inferencia por ventanas (offline) con cortes por pausa.
- WER esperable: ~26–30 en condiciones ideales, ~45 en YouTube variado ([05](../experiments/05_selftest_limpio.md)).
- Calibración validada en profundidad con un hablante (n=1); la generalización a más
  hablantes es trabajo pendiente.
- MPS requiere Apple Silicon; en CPU pura el total sube a ~1.5 s/segmento.
