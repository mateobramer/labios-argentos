# ENGINEERING_TP — el sistema de lectura de labios cerca de tiempo real

Guía del entregable de ingeniería. El doc técnico primario es [`SPEC.md`](SPEC.md)
(componentes, interfaces, latencias, decisiones justificadas); acá se organiza la
historia completa para la evaluación, con los gaps honestos.

## Qué es

Sistema **100 % local** que mira una boca por webcam (sin audio) y subtitula en vivo:

```
cámara 30 fps → MediaPipe FaceLandmarker (sticky-lock multi-cara)
             → VAD visual auto-calibrado (corta por pausas 0.45 s / tope 4 s)
             → crop de boca 96×96 (warp mean-face) → .npz
             → infer_server (ViSpeR 288M: encoder MPS ~0.17 s + beam 3 CPU ~0.9 s)
             → [opcional] qwen3:4b n-best rescoring (Ollama, +1.2 s, −3.0 WER)
             → UI web (SSE): subtítulo + tira de ROIs + guion acumulado
```

Detalle de componentes e interfaces (protocolo `CONFIG`/`READY` del infer_server,
endpoints HTTP): [SPEC §2-3](SPEC.md). Cómo correrla: [README](../README.md) §Demo,
`demo/README.md`.

## Latencia (medida, MacBook M1)

**~1.1 s por segmento sin corrector; ~2.3 s con corrector** — tabla por etapa en
[SPEC §4](SPEC.md), metodología y sweeps en [exp. 09](experiments/09_velocidad_inferencia.md).

## Optimizaciones probadas (y las descartadas)

Todas medidas en [exp. 09](experiments/09_velocidad_inferencia.md):

- ✅ **beam 3** — 2.2× más rápido que beam 40 con el mismo WER (sweep con IC).
- ✅ **encoder en MPS** — frontend 3.4× (0.57→0.17 s), transcripciones 100/100 idénticas.
- ❌ **int8** — Pareto-dominado en M1 (pierde WER y el LLM no lo recupera, §G).
- ❌ **CTC-greedy** — +12 WER, inaceptable.
- ❌ **todo en MPS** — espnet rompe por device mismatch (el beam queda en CPU por diseño).
- ❌ **qwen 9b / top-10 / scores en prompt** — no mejoran; 9b es 3.4× más lento ([04](experiments/04_qwen_corrector.md)).

Trade-off velocidad↔WER completo: [exp. 09](experiments/09_velocidad_inferencia.md);
decisiones consolidadas: [SPEC §5](SPEC.md).

## Robustez y fallbacks (estado real)

Implementado ([SPEC §6](SPEC.md)): fallback silencioso a 1-best si Ollama está apagado
o falla; warmup tolerante de cámara y MPS; validación de tomas en calibración
(≥20 frames, ≥60 % detección); sticky-lock con aviso en UI cuando hay >1 cara.

**Pendiente conocido** ([TO-DO](NEXT_STEPS.md) §4): reinicio automático si muere el
`infer_server` (hoy: error por línea), mensajes guiados para cámara sin permiso,
suite de tests automatizados del VAD/norm()/protocolo.

## Calibración al hablante (estado real)

Funciona end-to-end y está validada con **un** hablante ([exp. 10](experiments/10_adaptacion_hablante.md)):
UI `/calibrar` graba ~40 frases push-to-talk → `personalization/calibracion/calibrar_entrenar.sh`
entrena un **LoRA** (r16/α32) en una VM L4 spot (~10 min, ~$0.05, se autodestruye) →
el modelo personal baja el WER personal 29.2→24.5 **sin olvidar** el test general
(45.22→44.54). El full-FT con la receta del 50M **colapsa** el 288M (98.7 WER) — por
eso LoRA. Gap: generalización a más hablantes no validada (n=1).

## Captura de feedback humano (estado real)

- ✅ **Existe**: modo "Ayudanos a entrenar" en `/calibrar` — dona pares clip+texto
  leyendo frases sugeridas (quedan locales, no se versionan).
- ✅ **Existe (2026-07, mínimo)**: corrección humana de predicciones — botón ✏️ por
  segmento en el guion → `POST /feedback` → JSONL local en `data/feedback/` (gitignored)
  con predicción, corrección, timestamp, config del modelo e id de clip. Sin envío
  externo por diseño. Pendiente: usar esos pares para fine-tune/rescorer
  ([NEXT_STEPS](NEXT_STEPS.md) §3).

## Próximos pasos concretos para bajar latencia

(Detalle en [NEXT_STEPS](NEXT_STEPS.md) §2.) Los candidatos con mejor relación
esfuerzo/beneficio, en orden: (1) el beam search CPU es el 80 % del tiempo — probar
batch del beam o decoder alternativo; (2) pipeline overlap: inferir el segmento N
mientras se captura el N+1; (3) recorte del tope de 4 s por segmento cuando la pausa
llega antes. El encoder ya está optimizado (MPS).

## Limitaciones honestas

- **No es streaming causal.** El modelo es offline/bidireccional; el sistema aproxima
  tiempo real cortando por pausas. Un VSR causal cuadro a cuadro es otra arquitectura.
- WER ~26–30 en condiciones ideales; ~45 en YouTube variado ([exp. 05](experiments/05_selftest_limpio.md)).
- Calibración validada con n=1 hablante.
- MPS requiere Apple Silicon (en CPU el total sube a ~1.5 s/segmento).
- Los pesos base (`visper_vsr_base.pth`, 1.1 GB) no se versionan — obtención en
  [`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md).

## Costos

Inferencia $0 (local). Entrenamientos: VMs L4 spot ~$0.28/h → ~$1-3 por fine-tune del
50M, ~$0.05 por calibración personal ([SPEC §7](SPEC.md)).
