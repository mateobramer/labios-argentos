# Plan de tiempo real y panorama de benchmarks

Continuación de [`01_hallazgos_repo.md`](01_hallazgos_repo.md) (estudio del repo mpc001 y del
zero-shot ES sobre rioplatense). Este doc documenta el **pivote del proyecto hacia tiempo real
(streaming)** y ubica nuestros números dentro del panorama de benchmarks de VSR. Fecha: 2026-07-05.

---

## 1. Contexto y estado actual

El proyecto arrancó como un problema de **VSR (lectura de labios) en español rioplatense**:
tomar un modelo grande pre-entrenado, adaptarlo a nuestro acento con un dataset propio de YouTube,
y medir cuánto se degrada. Ese primer tramo ya está hecho.

**Dónde estamos parados:** tenemos un *teacher* fine-tuneado sobre rioplatense, `ft05`.

- `ft05` = base LIP-RTVE fine-tuneada con ~12k clips (~14h). Arquitectura de la familia
  Conv3D + ResNet18 → Conformer → decoder híbrido CTC/Attention.
- Es **offline / bidireccional**: el Conformer atiende a todo el contexto (pasado y futuro) y el
  decoder de atención decodifica por *beam search* hasta emitir `<eos>`. Necesita el clip entero.
- Resultado sobre nuestro test held-out (`test-658`, 2 hablantes f22/f15 nunca vistos en train):
  **WER 65.05 ± 1.29 / CER ~38–42**. Es el mejor número propio a la fecha.

El teacher funciona. El objetivo ahora es **llevarlo a tiempo real**: que el modelo lea labios
sobre un stream de cámara y muestre texto mientras la persona habla, sin esperar a que termine la
frase. Ese es el salto que documenta este archivo.

---

## 2. El diseño objetivo (del survey)

El PDF de diseño original (`survey-nlp/RESEARCH VISUAL SPEECH RECOGNITION NLP.pdf`) **no está
desactualizado**: ya anticipaba exactamente este destino. El pipeline objetivo que describe es:

```
[ CÁMARA ]  25 fps, RGB
      ↓
[ PREPROCESAMIENTO ]  MediaPipe (landmarks faciales) → crop región labial 96×96
                      → normalización, grayscale → chunks de 8 frames (= 320 ms)
      ↓  chunks de 8 frames
[ MODELO VSR ]  ResNet-3D (congelado) → CausalConformer (student) → CTC head (vocab español)
                destilado de Auto-AVSR, fine-tuneado en español, arquitectura CAUSAL (solo ve pasado)
      ↓  tokens crudos cada 320 ms
[ TEXTO ESPECULATIVO ]  acumula tokens; acumula chunks VSR hasta ~1.5 s de texto; commitea al LLM
      ↓  chunks acumulados
[ LLM CORRECTOR ]  qwen3:4b local (Ollama), system prompt = corrector de español rioplatense
                   input "vos tienes rason che" → output "vos tenés razón, che"; thread aparte
      ↓  chunk de texto commiteado
[ DISPLAY ]  texto commiteado+corregido, más cola especulativa cruda
```

Puntos clave del diseño, que siguen vigentes:

- **Arquitectura causal**: el CausalConformer nunca mira frames futuros → puede emitir tokens cada
  320 ms sin esperar el final de la frase. Es lo que habilita el streaming genuino.
- **Distillation**: student chico destilado del teacher grande (Auto-AVSR) para abaratar el forward
  pass y correr en vivo.
- **LLM corrector**: qwen3:4b local vía Ollama cada ~1.5 s, en un thread separado que no bloquea el
  flujo de tokens. Es el componente NLP/agéntico del proyecto.

La *research question* del survey: **"¿cuánto se degrada un modelo causal destilado de Auto-AVSR al
adaptarlo a rioplatense?"**, y las métricas encadenadas
`WER_baseline → WER_teacher → WER_student → WER_distillation → WER_llm`, más la parte de
eficiencia (params, RTF, latencia por chunk).

**Traducción a dónde estamos:** el survey define una cadena de modelos. Nosotros tenemos el
**teacher listo** (`ft05`, el `WER_teacher`). Falta todo lo causal/streaming (student, KD) y la
capa de corrección LLM. Este doc es el plan para recorrer lo que queda.

---

## 3. Panorama de benchmarks

Todos los sistemas *serios* del panorama son de la misma familia arquitectónica
(Conv3D + ResNet18 → Conformer → CTC/Attention) y todos son **offline / bidireccionales**. La
naturaleza de los datos y el acento explican casi toda la varianza de los números.

### 3.1. Sistemas serios (offline)

| Sistema | Params | Dataset (naturaleza) | Métrica | Modo |
|---|---|---|---|---|
| **Auto-AVSR** (teacher EN) | ~250M | LRS2 (TV BBC) | 14.6% WER | offline |
| **Auto-AVSR** (teacher EN) | ~250M | LRS3 (charlas TED), 3448h | 19.1% WER | offline |
| **Auto-AVSR** (teacher EN) | ~250M | LRS3 (charlas TED), solo 818h | 36.3% WER | offline |
| **mpc001 multilingüe** | ~50M | LRS3 (TED) | 19.1% WER | offline |
| **mpc001 multilingüe** | ~50M | LRS2 (TV BBC) | 26.1% WER | offline |
| **mpc001 multilingüe** | ~50M | CMLR (noticiero TV chino) | 8.0% CER | offline |
| **mpc001 multilingüe** | ~50M | CMU-MOSEAS **ES** (YouTube) | 43.9–44.5% WER | offline |
| **mpc001 multilingüe** | ~50M | CMU-MOSEAS PT (YouTube) | 51.4% WER | offline |
| **mpc001 multilingüe** | ~50M | CMU-MOSEAS FR (YouTube) | 58.6% WER | offline |
| **Gimeno** (español) | ~102M (incl. LM) | VLRF (lab, 24 hab, 3h) | 24.8% WER (spk-dep) | offline |
| **Gimeno** (español) | ~102M (incl. LM) | LIP-RTVE (noticiero RTVE, 323 hab, 13h) | 34.5% (spk-dep) / **59.5% (spk-indep)** | offline |
| **Gimeno** (español) | ~102M (incl. LM) | CMU-MOSEAS ES (YouTube) | 44.6% WER | offline |
| **Gimeno** (español) | ~102M (incl. LM) | MuAViC ES | 56.3% WER | offline |

### 3.2. Sistemas de tiempo real / demos

| Sistema | Datos / checkpoint | Métrica | Modalidad | Modo real |
|---|---|---|---|---|
| **Chaplin** (amanvirparhar/chaplin) | checkpoint Auto-AVSR LRS3 | 19.1% WER (heredado) | visual-only | **push-to-talk** (no streaming): grabá, soltá, inferencia offline sobre el clip entero. MediaPipe + corrección Qwen3-4B (Ollama) |
| **PyTorch "Real-time AV-ASR"** (blog) | Emformer + RNN-T, LRS3 | 1.6–2.6% WER; RTF 0.87 CPU / 0.33 GPU; latencia algorítmica 800 ms | **audio-visual** | streaming real, pero NO visual-only y sin checkpoint visual publicado |

Chaplin es literalmente el diseño del PDF ya construido... salvo que **no es streaming**: es
push-to-talk. La latencia es "duración del clip + inferencia + LLM". Sirve como *blueprint* de
ingeniería (MediaPipe + buffer + Qwen3-4B), no como sistema en vivo.

El único streaming real del ecosistema (PyTorch AV-ASR) es **audio-visual**: se apoya en el audio
para bajar el WER a rango 1–3%. No hay checkpoint visual-only que reutilizar.

### 3.3. Juguetes (mencionados y descartados)

- **LIP-TRAC**: prototipo estudiantil, sin repo, CRNN, "<33% WER" (marketing), 6.3 s de latencia en
  Raspberry Pi 5.
- **"Deep Learning Lip-Reading for Vocal Impaired"**: 25 palabras en italiano, laboratorio, 96.4%
  accuracy a nivel palabra (no oracional).
- **"Visual Voice"**: GRID, 97.6% accuracy, demo.

Ninguno aporta para español continuo/oracional ni para streaming.

### 3.4. Nuestros números

| Modelo | Base | Datos FT | %WER | %CER | Notas |
|---|---|---|---|---|---|
| **ft05** (mejor propio) | LIP-RTVE | ~12k clips (~14h) | **65.05 ± 1.29** | 38.24 ± 0.97 | offline/bidireccional, sin LM externo |
| zero-shot mpc001 ES | CMU-MOSEAS | 0 | 71.50 ± 1.70 | 46.88 ± 1.24 | con beam=30 + RNNLM y todo |
| ft05b (A/B same-data) | LIP-RTVE | 8067 clips | 70.30 ± 1.30 | 42.08 ± 0.95 | mismo train que ft07 |
| ft07 (A/B same-data) | multilingüe (CMU-MOSEAS ES) | 8067 clips | 69.15 ± 1.28 | 41.66 ± 0.91 | mismo train que ft05b |

Lecturas del A/B `ft05b` vs `ft07` (mismo train de 8067, mismo seed): con datos idénticos la base
multilingüe transfiere **apenas** mejor al rioplatense (−1.16 WER), pero los IC95% se solapan →
diferencia **no significativa**. Y el costo de recortar datos es grande: `ft05b` (8067 clips)
rinde 70.30 vs `ft05` (10934 clips) 65.05 = **−5 WER solo por tener 26% más de datos**. Conclusión
dura: **más datos importa mucho más que qué base elegir.**

---

## 4. Análisis técnico central

### 4.1. "Más rápido" ≠ "en vivo"

Es el malentendido central que hay que evitar. Distilar y quantizar **abaratan el forward pass**,
pero no convierten un modelo en streameable. `ft05` es bidireccional por dos razones estructurales:

1. el **Conformer atiende al futuro** (self-attention sobre toda la secuencia), y
2. el **decoder de atención** decodifica por beam search hasta emitir `<eos>`, o sea necesita saber
   dónde termina la frase.

Un modelo bidireccional **no se puede streamear por más chico que sea**. Achicarlo lo hace más
rápido, no incremental. Para streaming hay dos caminos, con costos muy distintos:

- **(a) Sliding-window sobre el modelo offline** — pragmático, sin re-entrenar. Se corre el modelo
  offline repetidamente sobre una ventana deslizante del stream. Es lo que hace (más o menos)
  Chaplin. Barato, imperfecto (recomputa, sufre en los bordes de ventana), pero factible ya.
- **(b) Encoder causal + decode CTC** — el camino "de verdad". Encoder que solo mira el pasado +
  decodificación CTC (frame-síncrona, monótona, sin `<eos>`), que emite token por frame. Es
  streameable de nacimiento, pero **requiere re-entrenar** (es research).

### 4.2. Palancas ordenadas por ROI

De lo más barato/seguro a lo más caro/riesgoso:

1. **Decoding-side, gratis**: bajar el beam width, usar CTC-greedy, sacar el LM externo. No cuesta
   entrenamiento, se mide de una. Es el primer barrido que hay que hacer.
2. **Quantización** INT8/fp16: acelera el forward sin re-entrenar. Bajo riesgo.
3. **Conversión a causal**: re-entrenar el encoder como causal (camino (b)). Costo alto (GPU, datos),
   pero es el que habilita streaming genuino.
4. **Distillation a un student chico**: el *lift* más grande de eficiencia... y **el más riesgoso**
   con nuestros ~14h de datos. Destilar necesita datos; con tan poco, el student puede degradar feo.

### 4.3. Hallazgo estratégico: nadie tiene VSR visual-only en streaming

La búsqueda en el ecosistema confirma un hueco: **no existe un sistema de VSR visual-only en
streaming**. El único streaming real (PyTorch AV-ASR) usa audio. Todo lo visual-only serio es
offline. Los "en vivo" (Chaplin) son en realidad push-to-talk sobre inferencia offline.

Doble implicancia:

- El **student causal** (camino (b)) sería una **contribución genuina** — y justamente por eso es
  difícil (nadie lo tiene resuelto, no hay checkpoint que reusar).
- El atajo **sliding-window / push-to-talk** (tipo Chaplin) es lo que efectivamente existe hoy, y es
  el camino pragmático para una demo en vivo sin research.

### 4.4. Los datos y el acento dominan

El panorama grita una sola cosa: el WER lo manda el dominio y el acento de los datos, no la
arquitectura. LRS3 en inglés da 19.1%; el mismo tipo de modelo sobre CMU-MOSEAS en español da
~44.5%; y sobre rioplatense zero-shot sube a 71.5. Y dentro del mismo idioma: Auto-AVSR pasa de
19.1% (3448h) a 36.3% (818h) solo por tener menos horas. **Nosotros estamos en ~14h.** Cualquier
plan tiene que asumir que el techo lo pone la cantidad y calidad de datos rioplatenses, no el
tuneo del modelo.

---

## 5. El corrector LLM y la hipótesis del CER

El corrector LLM no es decorado: **es el componente agéntico / NLP que el proyecto requiere**
(que el sistema no sea "solo la cáscara" de un modelo de visión). Toma texto crudo del VSR y lo
corrige a español rioplatense bien escrito.

**Hipótesis central (intuición confirmada teóricamente):** para corrección con LLM importa **más un
CER bajo que un WER bajo**.

- Si el **CER es bajo**, la string cruda conserva señal carácter a carácter aunque tenga palabras
  mal — el LLM tiene material para reconstruir ("vos tienes rason che" → "vos tenés razón, che").
- Si el **CER es alto**, la string es ruido — el LLM **alucina** y puede **empeorar el WER**
  inventando texto plausible pero incorrecto.

Nuestro CER es ~42%, que es **alto**. Por eso la ganancia neta del corrector es una **pregunta
empírica**: no sabemos a priori si el LLM ayuda o perjudica en este régimen. Hay que medirlo.

**Diseño del corrector:**

- Pasar **n-best** (no solo el 1-best) — más hipótesis = más señal para el LLM.
- Pasar **contexto** (el caption previo ya commiteado) para coherencia.
- Aplicar solo a **utterances ya commiteadas** (no cada 320 ms), en un **thread aparte** que no
  bloquea el flujo especulativo.
- **qwen3:4b local** vía Ollama por latencia (una API tipo Claude daría techo de calidad, pero
  agrega latencia de red; el local es la apuesta de latencia segura).

---

## 6. Plan por fases

Ordenado por ROI. La recomendación es arrancar por la Fase 0.

### Fase 0 — Corrección LLM offline (en curso, gratis, sin GPU)

Experimento offline: correr el corrector LLM sobre los archivos `test.inf` que **ya tenemos** de los
658 clips (pares ref/hyp de `ft05b`/`ft07` ya descargados) → medir `WER_llm`. No necesita cámara ni
GPU. Doble valor: (1) resuelve el requisito del componente agéntico, y (2) **zanja empíricamente lo
del CER** — decide si con CER 42% el LLM ayuda o perjudica, antes de invertir en bajar CER.
**Estado: en curso, se eligió qwen3:4b local vía Ollama.**

### Fase 1 — Harness de benchmark de inferencia

Medir eficiencia de verdad: **RTF + latencia por chunk + time-to-first-token**, desglosado por etapa
(frontend / encoder / decoder). Barrido de configuraciones: `{beam+LM, CTC-greedy} × {fp32, INT8/fp16}
× {CPU, GPU} × {clip entero, chunked}`, reportando WER/CER de cada combo. Es lo que llena la fila de
"Eficiencia" del survey y cuantifica cuánto rinde cada palanca decoding/quantización.

### Fase 2 — Demo en vivo sliding-window

Demo en vivo aplicando sliding-window sobre `ft05` (sin re-entrenar), blueprint = Chaplin:
webcam → MediaPipe → chunks → captions. El trabajo grande acá **no es el modelo**, es el **frontend
en vivo**: MediaPipe → alineación a la cara media → crop 96×96 en tiempo real, empatando exactamente
la convención de preproc con la que se entrenó.

### Fase 3 — Student causal ± KD (research, si hay tiempo + GPU)

El camino (b): encoder causal, opcionalmente con knowledge distillation desde `ft05` como teacher.
Da los números `WER_student` / `WER_distillation` del survey y sería la contribución de research.
**Riesgoso por la escasez de datos (~14h)** — destilar/entrenar causal con tan poco es la parte más
incierta del plan.

**Recomendación:** empezar por **Fase 0**. Es barata, resuelve el requisito agéntico, y decide si
conviene invertir en bajar el CER antes que el WER. Modelos e inferencias disponibles:
`ft05b_best.pth` / `ft07_best.pth` + sus `test.inf`/`test.wer` en `~/Desktop/labios-argentos/modelos/`
y `multilingual-vsr/runs/`.

---

## 7. Fuentes

- **Auto-AVSR** — "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels",
  arXiv:2303.14307.
- **mpc001 multilingüe** — "Visual Speech Recognition for Multiple Languages", Ma et al.,
  arXiv:2202.13084. Repo: github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages.
- **Gimeno (español)** — "Factors of Influence on End-to-End Continuous Spanish Lipreading",
  arXiv:2502.00464.
- **Chaplin** — github.com/amanvirparhar/chaplin.
- **PyTorch "Real-time AV-ASR"** — blog de PyTorch sobre AV-ASR en tiempo real (Emformer + RNN-T).
