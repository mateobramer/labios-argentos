# 06 — Demo tiempo real (push-to-talk) + remap de ft05

## A. Demo push-to-talk (funciona, local)

Objetivo: grabarse, apretar barra espaciadora, hablar, y que el modelo tire la transcripción (estilo
Chaplin: push-to-talk, inferencia offline sobre el clip).

**Arquitectura:** `demo/demo_ptt.py` (env `ptt`: cv2 + MediaPipe) abre cámara → ESPACIO graba/corta
(con **debounce 0.6s**, sino un tap togglea varias veces) → crop MediaPipe mean-face 96×96 → npz →
`demo/infer_server.py` (env `visper`, ViSpeR cargado en memoria) devuelve el texto.

**Modelo:** ViSpeR (nuestro mejor, 45 WER; en limpio 23). Corre en **CPU** (~6-8s/clip); **NO usar MPS**
(rompe el beam de espnet por device mismatch). Arranque ~7-40s (carga 1.1GB).

**Validado end-to-end** sobre un clip: "tengo un tripode que lo hice con cinta..." → "si me tengo un tripo
que... con cinta... el esfuerzo...". Lip-reading imperfecto pero capta lo central.

**Aprendizajes de macOS:** (1) permiso de cámara para la terminal; (2) la ventana cv2 debe tener foco para
tomar la barra; (3) el **primer `cap.read()` suele fallar** mientras la cámara calienta → hace falta warmup
+ tolerar fallos transitorios (sino el loop corta con 0 frames).

## B. Remap ft05 espnet2 → espnet1 (para correr el 50M local)

**Problema:** el env de Gimeno (`vsr-factors`, espnet2) no está local y su `ASRTask` arrastra todo el zoo
de decoders (s4/whisper/hf) → difícil de instalar en M1. **Solución:** cargar ft05 en el env `mvsr`
(espnet1 vendoreado de mpc001), que SÍ funciona y decodifica.

**Cómo:** el mapa espnet2→espnet1 se deriva **por posición** comparando `es_remapped.pth` (espnet2) vs el
`model.pth` de CMU-MOSEAS (espnet1) — son 767 tensores idénticos por valor y en el mismo orden (rename
puro). Se aplica ese mapa **por nombre** a `ft05_best.pth` (mismas claves que es_remapped) →
`modelos/ft05_espnet1.pth`. Carga en `pipelines.model.AVSR` con la config `CMUMOSEAS_V_ES_WER44.5`
(mismo char_list 37) y decodifica bien (validado: "que buena onda espero que la esten disfrutando como yo"
→ "que es buena onda pero que haya disfrutado como yo").

Esto habilitó correr ft05 local (n-best incluido) sin GPU ni el env pesado de Gimeno.

## Estado / próximo
- **Push-to-talk** (`demo_ptt.py`): funciona. El server ahora corre **encoder en MPS + beam 3** (~1.1 s/clip;
  antes ~6-8 s) y soporta **qwen n-best** con `VSR_QWEN=1` (ver [09](09_velocidad_inferencia.md)).
- **Streaming** (`demo_stream.py`): implementado — **VAD visual** (corta segmentos en pausas de labios,
  auto-calibrado con 2 s de silencio al arrancar), landmarks en vivo durante la captura, **transcript
  acumulado** con dedup de bordes. Flags: `--pause`, `--max-seg`, `--sens`, `--qwen`. Es una aproximación
  (ViSpeR sigue siendo offline/bidireccional); el streaming causal real requiere el student (memoria
  `realtime-vsr-plan`).
- **UI web** (`demo_web.py` + `web/index.html`): mismo motor que el streaming pero servido en
  `http://localhost:8551` — cámara MJPEG, chip de estado (calibrando/escuchando/hablando/leyendo),
  medidor de labios con umbral, captions grandes, panel de transcript con duración+latencia por segmento,
  copiar/limpiar. Solo stdlib (sin deps nuevas); renderiza ñ/acentos (cv2 no podía). El server de
  inferencia manda una línea `CONFIG {json}` antes del `READY` (los clientes viejos la ignoran).
- **Calibración al hablante integrada** (`/calibrar` + `demo/calibracion/`): la persona graba ~40 frases
  (push-to-talk en el browser, guarda npz+txt en `~/vsr_personal/<nombre>/`), después
  `bash demo/calibracion/calibrar_entrenar.sh <nombre>` entrena el LoRA en una L4 spot (~10 min, ~$0.05,
  auto-destruye) y baja `modelos/personal/<nombre>.pth` → `demo_web.py --ckpt ...` (server: `VSR_CKPT`).
  Método validado en [10](10_adaptacion_hablante.md). La misma página tiene el modo **"Ayudanos a
  entrenar"**: texto libre (decís algo + escribís qué dijiste) → dona pares clip+texto a `~/vsr_contrib/`.
- **Toggle de qwen en runtime**: botón en el header (protocolo `::qwen 0|1` por stdin al server; el server
  corre beam 5 fijo para tener las 5 candidatas disponibles al togglear).
- **Multi-cara (demos en vivo)**: el landmarker detecta hasta 3 caras y sigue con **sticky-lock** a la más
  grande (la más cercana); si hay >1 cara, dibuja el recuadro "leyendo a esta persona" en el video y avisa
  en el header. Evita que el crop salte entre personas en un ambiente con público.
