# 09 — Velocidad de inferencia: benchmark y aceleración

**Objetivo:** medir cuánto tarda la inferencia de nuestros dos modelos (ViSpeR y el mejor ft de Gimeno,
ft05), cómo se degrada (o no) la performance al acelerarlos, y qué técnicas de aceleración son viables
para llevar esto a tiempo real. Todo sobre **mis 40 clips del self-test** (mismos de [05](05_selftest_limpio.md)).

## Setup

- **Hardware:** Apple M1, 8 cores, 16 GB. **CPU-only** (no hay CUDA; MPS rompe el beam-search de espnet
  por device-mismatch → se corre en CPU, igual que el zero-shot y la demo).
- **Datos:** 40 clips, **126.2 s de video total** (frases de frente, buena luz), 25 fps, 65–99 frames/clip
  (2.6–4.0 s). Ground-truth = las frases que leí.
- **Métrica:** misma `norm()` + WER/CER que todos los docs. **Tiempo:** `time.perf_counter` por clip,
  con un **warmup descartado** (la 1ª inferencia incluye lazy-init/caches). `torch` threads=4 (default M1).
- **RTF** (real-time factor) = tiempo_inferencia / duración_clip. **RTF < 1 = más rápido que tiempo real.**
- Scripts: scratchpad `bench_speed.py` (una config por corrida → JSON) + `run_bench.sh`.

## A. Arquitectura: por qué cada uno tarda lo que tarda

| | **ViSpeR** (nuestro mejor) | **ft05** (50M, Gimeno) |
|---|---|---|
| Params | ~288M | ~50M |
| Frontend | Conv3D + ResNet18 | Conv3D + ResNet18 |
| Encoder | Conformer ×12, **adim 768**, 12 heads, FFN 3072 | Conformer ×12, **adim 256**, 4 heads, FFN 2048 |
| Decoder | Transformer ×6 (autoregresivo) | Transformer ×6 (autoregresivo) |
| Vocab | unigram SPM (~subwords) | 37 chars |
| Decoding | híbrido CTC/attn, **beam 40** | híbrido CTC/attn + LM, **beam 30** |

**Dónde está el costo en CPU:** el frontend (conv3D+ResNet) corre **una vez** por clip; el **beam-search es
autoregresivo** → re-evalúa el decoder Transformer `beam × longitud` veces. En la literatura, el beam-search
en CPU es **~90% del runtime** de estos modelos. → **el beam-width es la palanca #1.**

## B. Baseline (config de fábrica) — WER/CER + tiempo

| Modelo | beam | %WER | %CER | s/clip (mean) | RTF | tiempo total 40 clips |
|---|---|---|---|---|---|---|
| **ViSpeR** | 40 | **23.60** | **11.41** | 3.17 | 1.00 | 127 s |
| ft05 (50M) | 30 | 52.81 | 29.62 | 7.17 | 2.27 | 287 s |

**ViSpeR domina en los DOS ejes: es ~2.3× más rápido Y menos de la mitad de WER.** El ft05, pese a ser 6×
más chico, es **más lento** porque igual corre beam 30 + un LM externo, y encima decodifica peor.
(El WER de ft05 acá, 52.8, es mejor que el 68 histórico del test-658 porque el self-test es limpio; ver [05](05_selftest_limpio.md).)

## C. Beam-width sweep (la palanca #1)

### ViSpeR

| beam | %WER | %CER | s/clip | RTF | speedup vs b40 |
|---|---|---|---|---|---|
| **40** (baseline) | 23.60 | 11.41 | 3.17 | 1.00 | 1.0× |
| 10 | 24.16 | 11.75 | 1.70 | 0.54 | 1.9× |
| 5 | 23.60 | 11.80 | 1.50 | 0.48 | 2.1× |
| **3** ⭐ | **23.31** | 11.52 | 1.47 | 0.47 | **2.2×** |
| 1 (greedy) | 25.84 | 12.59 | 0.84 | 0.27 | 3.8× |

**Bajar el beam de 40 a 3 es GRATIS**: 2.2× más rápido y el WER **no empeora** (23.60 → 23.31, incluso baja
un pelo dentro del ruido). El beam=40 default es **puro desperdicio de cómputo**. Incluso el greedy (beam=1)
cuesta apenas **+2.2 WER / +1.2 CER** a cambio de **3.8× de velocidad** (0.84 s/clip, RTF 0.27 → 4× más
rápido que tiempo real). Solo 14/40 clips cambian entre beam40 y beam1, y a veces beam1 **acierta más**
(ver §F, clip_06).

### ft05 (contraste)

| beam | %WER | %CER | s/clip | RTF |
|---|---|---|---|---|
| 30 | 52.81 | 29.62 | 7.17 | 2.27 |
| 5 | 55.62 | 30.92 | 4.47 | 1.41 |
| 1 | 79.49 | 46.77 | 1.41 | 0.45 |

**ft05 SÍ se derrumba** al bajar el beam (52.8 → 79.5 WER en greedy). El modelo débil **depende del beam
ancho + el LM** para compensar; ViSpeR es lo bastante bueno como para no necesitarlo. → otra razón para
quedarnos con ViSpeR: **tolera la aceleración**.

### ¿Por qué ft05 (50M) es MÁS LENTO que ViSpeR (288M) si es 6× más chico?

Porque el tiempo **no lo domina el tamaño del modelo** (el encoder corre 1 vez y es barato), sino el
**bucle autoregresivo de beam-search**, que ft05 corre muchas más veces. Descompuesto empíricamente:

| causa | efecto | medición |
|---|---|---|
| **LM externo** (ft05 usa uno; ViSpeR no) | forward extra del LM en **cada paso × cada beam** | ft05 beam30 **con LM 7.59 s → sin LM 2.88 s** (el LM = ~62% del tiempo) |
| **vocab char-level** (37 chars) vs **subword** (ViSpeR) | ft05 decodifica letra por letra → muchos más pasos | **44 pasos/frase (ft05) vs 12 (ViSpeR) = 3.8× más pasos** |

Con un modelo chico, cada paso del decoder es barato en matmuls, pero el **overhead fijo por paso**
(bookkeeping del beam, scorer CTC, y sobre todo el LM) domina. ft05 hace 3.8× más pasos y encima llama al
LM en cada uno → termina más lento que ViSpeR pese a tener 6× menos parámetros. Sin LM, ft05 (2.88 s) ya se
acerca a ViSpeR beam40 (3.17 s), confirmando que **el LM es el culpable principal**.

## D. Barrido extendido: frontera de Pareto (beam × ctc_weight × int8 × CTC-greedy)

Barrido completo sobre ViSpeR variando y combinando todas las palancas, con **IC 95% bootstrap** (mismo
`norm()`), **mediana** de tiempo (robusta), warmup descartado, `OMP_NUM_THREADS=4`, cooldown 25 s entre
configs y **centinela de deriva** (beam5 al inicio y al final: −9.8%, o sea el final salió *más rápido* →
**sin thermal throttling**; WER idéntico 23.6/23.6 = determinismo confirmado).

Ordenado de más rápido a más lento; **speedup vs beam40**:

| config | %WER | ±IC | %CER | ±IC | med s/clip | RTF | speedup |
|---|---|---|---|---|---|---|---|
| **encoder-only** (piso teórico) | — | | — | | **0.571** | 0.18 | **5.6×** |
| CTC-greedy | 35.39 | 7.1 | 18.38 | 4.3 | 0.627 | 0.20 | 5.1× |
| CTC-greedy + int8 | 37.92 | 7.3 | 19.28 | 4.1 | 0.631 | 0.20 | 5.1× |
| beam 1 (greedy) | 25.84 | 7.8 | 12.59 | 4.2 | 0.835 | 0.27 | 3.8× |
| beam 1 + int8 | 26.69 | 7.7 | 13.43 | 4.4 | 0.836 | 0.27 | 3.8× |
| beam 2 | 23.88 | 6.8 | 11.69 | 3.8 | 1.164 | 0.39 | 2.8× |
| **beam 3** ⭐ | 23.31 | 6.6 | 11.52 | 3.7 | 1.431 | 0.47 | **2.2×** |
| beam 5 | 23.60 | 6.6 | 11.80 | 3.7 | 1.574 | 0.53 | 2.0× |
| beam 5 + int8 | 24.72 | 6.8 | 12.20 | 3.9 | 1.338 | 0.43 | 2.4× |
| beam 5 · ctc 0.3 | 26.97 | 6.8 | 12.82 | 3.6 | 1.373 | 0.44 | 2.3× |
| beam 5 · ctc 0.5 | 28.09 | 7.3 | 14.00 | 4.0 | 1.379 | 0.44 | 2.3× |
| beam 5 · ctc 1.0 | 38.76 | 7.7 | 19.67 | 4.5 | 1.985 | 0.65 | 1.6× |
| beam 10 | 24.16 | 6.7 | 11.75 | 3.7 | 1.737 | 0.54 | 1.8× |
| beam 10 + int8 | 26.40 | 7.1 | 12.48 | 3.7 | 1.710 | 0.56 | 1.9× |
| beam 20 | 24.16 | 6.7 | 11.58 | 3.6 | 2.182 | 0.68 | 1.5× |
| **beam 40** (baseline) | 23.60 | 6.6 | 11.41 | 3.5 | 3.213 | 1.00 | 1.0× |

### Lecturas (qué es señal y qué es ruido)

1. **Todo el eje de beam ≥ 2 es un PLATEAU plano en accuracy.** beam 2/3/5/10/20/40 dan WER 23.3–24.2, todos
   **dentro del mismo IC (±6.6)** → estadísticamente **indistinguibles**. El beam=40 default es puro
   desperdicio: **beam 3 da el mismo WER a 2.2× la velocidad** (RTF 0.47). Es el punto recomendado.
2. **La frontera de Pareto es SOLO el eje beam.** Ninguna otra palanca mejora sobre "elegí el beam justo":
   - **int8 está Pareto-dominado.** Agrega ~+1 WER y da poca o nula velocidad: en beam1 el speedup es
     **cero** (0.835→0.836 s, no hay Linear que aprovechar en greedy) y en beam5 baja solo a 1.338 s —
     pero **beam3 fp32 (1.431 s, WER 23.3) le gana en las dos cosas** (más rápido de facto y mejor WER que
     beam5+int8). En M1/qnnpack no vale; en servidor x86 (fbgemm) la historia podría cambiar.
   - **Subir ctc_weight es estrictamente PEOR:** 0.1→0.3→0.5→1.0 empeora el WER (23.6→27→28→39) **y encima
     es más lento** (el CTC prefix-scoring cuesta). El default 0.1 es el correcto; no tocarlo.
3. **CTC-greedy: 5.1× más rápido pero cuesta caro (+12 WER, sí supera el IC → degradación REAL).** Como
   anticipamos, la rama CTC de ViSpeR fue objetivo auxiliar (`mtlalpha≈0.1`), no es un decoder de primera.
   Nota: el CER sube menos que el WER (11.4→18.4) — los errores son *typos* ("grfe"→jefe, "pataagoia") más
   que palabras enteras mal.
4. **El encoder es el PISO: 0.571 s (RTF 0.18).** CTC-greedy (0.627 s) está prácticamente pegado a ese piso
   → **el decode ya es casi gratis; el muro es el frontend conv3D+ResNet.** Implicación clave para tiempo
   real: por debajo de ~0.6 s/clip **no se baja tocando el decoding**; hay que atacar la arquitectura del
   frontend (SwinLip, destilación, menos capas ResNet). Todo lo demás ya está exprimido.

### ⭐ Encoder en MPS (híbrido GPU/CPU) — EL MURO DEL FRONTEND CAYÓ

El beam de espnet rompe en MPS (device-mismatch), pero el **encoder solo** es un forward puro → híbrido:
**encoder en MPS, features a CPU, beam en CPU** (con `PYTORCH_ENABLE_MPS_FALLBACK=1`). Sobre los 100 clips:

| (mediana, n=100) | CPU | **encoder MPS** | speedup |
|---|---|---|---|
| encoder-only | 0.583 s | **0.170 s** | **3.4×** |
| e2e beam3 | 1.386 s | **1.094 s** | 1.3× |
| transcripciones idénticas | — | **100/100** ✅ | |

- El piso del frontend pasó de 0.57 → **0.17 s**; ahora el cuello volvió a ser el **beam** (~0.9 s del e2e).
- Combos nuevos posibles: **MPS + beam1 ≈ 0.43 s** · **MPS + CTC-greedy ≈ 0.22 s (RTF 0.07)**.
- **Cableado en `demo/infer_server.py`** (auto-detecta MPS; `VSR_MPS=0` lo apaga). Validado end-to-end.
- Caveat: MPS compila kernels **por shape** (cada largo de clip nuevo) → los primeros clips tardan más
  (11 s → 2 s → 1.4 s) y después queda rápido. En uso sostenido no molesta.

### Nº de threads (barrido aparte, beam 3)

`torch.set_num_threads` ∈ {1,2,3,4,6,8}. **4 es el óptimo** (mediana 1.28 s); baja de 1→4 y **empeora en
6–8** (la M1 tiene 4 cores de *performance*; más allá desborda a los de *eficiencia*, más lentos, + overhead
de scheduling). WER idéntico (23.31) en todos → los threads no afectan el output, solo el tiempo. El default
que usamos (4) ya era el correcto.

| threads | 1 | 2 | 3 | **4** | 6 | 8 |
|---|---|---|---|---|---|---|
| med s/clip | 1.55 | 1.45 | 1.36 | **1.28** | 1.55 | 1.66 |

> **Ruido de timing:** hay **~±10% run-to-run** (beam3 dio 1.43 s en el barrido §D y 1.28 s acá, en corridas
> distintas). Los **ratios entre configs son más confiables que los s/clip absolutos**; las conclusiones de
> §D se sostienen porque los efectos reales (beam, CTC-greedy, encoder floor) son mucho mayores que ese ±10%.

## E. Tiempo vs. longitud del clip

El tiempo escala **~lineal con los frames** (el beam re-corre el decoder por cada paso de salida):

| Modelo (beam) | clip corto (65 fr / 2.6 s) | clip largo (99 fr / 4.0 s) |
|---|---|---|
| ViSpeR b40 | 2.50 s | 4.05 s |
| ViSpeR b1 | 0.74 s | 1.05 s |
| ft05 b30 | 6.57 s | 9.15 s |

→ El **RTF es aproximadamente constante** dentro de este rango, así que las cifras s/clip y RTF de arriba
son representativas para frases de ~3 s. Para frases más largas el tiempo sube proporcional (relevante para
streaming: una ventana fija acota el costo).

## F. Ejemplos (output vs. ground-truth)

`REF` = lo que dije · `V40`/`V1` = ViSpeR beam40 / beam1 · `F30` = ft05 beam30.

```
[clip_00 · 3.96s] REF: hoy me levante temprano y me fui a laburar en bici
            V40: hoy me levante temprano y me fui a laburar en bici        ✅ perfecto
            V1 : hoy me levante temprano y me fui a la urana en bici       (greedy pierde "laburar")
            F30: hoy me levante temprano y me fui a la guerra empresa

[clip_06 · 3.76s] REF: vamos a comer un asado el domingo en casa de mi vieja
            V40: vamos a comer un hasta dos domingos en casa de mi vieja
            V1 : vamos a comer un asado el domingo en casa de mi vieja     ✅ ¡beam1 ACIERTA y beam40 no!
            F30: vamos a lo mejor casado domingo casa de mi vida

[clip_08 · 3.40s] REF: no tengo ganas de cocinar pidamos una pizza
            V40: no te nos ganas de cocinar veamos una pizza               (confusión de visemas)
            F30: no tengo ganas de contar veamos una especie

[clip_09 · 3.16s] REF: mi hermano se compro un auto usado la semana pasada
            V40: mi hermano se compro un autocusado la semana pasada       ("auto usado"→"autocusado")
            F30: mi hermano se lo vio causado la semana pasada

[clip_36 · 3.16s] REF: plantamos unos tomates y albahaca en el balcon
            V40: planteamos una toma tan salvaje en el valor               (frase difícil, ambos fallan)
            F30: inventamos un costo para estar en el barco
```

Patrón: los errores de ViSpeR son **sustituciones por palabras válidas parecidas en labios** (viseme
confusion), no ruido aleatorio → es exactamente el régimen donde el **n-best rescoring** ayuda (doc [04](04_qwen_corrector.md) §F).

## G. Latencia del corrector LLM (qwen n-best rescoring)

El n-best rescoring (doc [04](04_qwen_corrector.md) §F) agrega una llamada a **qwen3:4b-instruct-2507-q4_K_M**
(Ollama local, `think=false`, temp 0) que recibe las **top-5 candidatas** del beam y devuelve una frase.
Esto se suma **encima** del tiempo de ViSpeR. Medido aislado sobre los mismos 40 clips:

| métrica | valor |
|---|---|
| latencia **caliente** por clip (wall) | **1.24 s** (mediana 1.22, min 0.90, max 1.69) |
| desglose Ollama (warm) | prompt_eval 0.42 s + generación 0.71 s |
| tokens | ~159 in / ~13 out · throughput **~18.4 tok/s** |
| **1ª llamada (fría, carga el modelo 2.5 GB)** | **3.70 s** (de los cuales 1.72 s es load) |

**Cuánto agrega:** el rescoring suma **~1.24 s por clip** encima de ViSpeR. En contexto, sobre la config
recomendada (ViSpeR beam 5):

| pipeline | s/clip | RTF | %WER |
|---|---|---|---|
| ViSpeR beam5 solo | 1.50 | 0.48 | 23.60 |
| **ViSpeR beam5 + qwen n-best** | **~2.74** | **0.87** | **20.22** (−3.4, doc 04 §F) |

**Casi duplica la latencia** (de 1.5 a 2.7 s) para bajar −3.4 WER. Sigue **por debajo de tiempo real** (RTF
0.87) en frases de ~3 s. Detalles a tener en cuenta:
- El costo es **~99% generación de tokens** (18 tok/s en la M1); la salida es corta (~13 tokens) así que es
  acotado, pero es secuencial y no se paraleliza con ViSpeR (primero decodifica el beam, después el LLM).
- **Primera invocación paga +3.7 s** por cargar el modelo; con `keep_alive=10m` queda caliente → en una demo
  o server que ya lo tiene en memoria, la latencia real es la caliente (1.24 s).
- El rescoring **necesita beam ≥ 5** (5 candidatas) → pone un piso al recorte de beam: no se puede combinar
  con greedy (beam=1). Para tiempo real hay un trade-off: beam=1 sin rescoring (0.84 s, WER 25.8) vs
  beam=5 + rescoring (2.74 s, WER 20.2).
- **qwen3.5:9b** (también instalado) sería más lento (modelo ~2.6× más grande) — no medido; el 4b ya da la
  mejora, el 9b es a explorar solo si sube la calidad del rescoring.

### A/B de pulido del n-best (n=100, IC pareado vs 1-best 29.51)

¿Se puede exprimir más el rescoring? Probamos 3 variantes contra el baseline:

| variante | %WER | delta | IC95 pareado | s/clip LLM |
|---|---|---|---|---|
| **A: top-5, qwen4b, prompt plano** (actual) | **26.46** | −3.04 | [+0.71,+5.53] ✅ | ~1.2 |
| B: top-10, qwen4b | 26.35 | −3.16 | [+0.82,+5.56] ✅ | 1.4 |
| C: top-5 + scores del beam en el prompt | 26.81 | −2.69 | [+0.47,+5.05] ✅ | 1.2 |
| D: top-5, **qwen3.5:9b** | 27.52 | −1.99 | [−0.47,+4.59] ❌ no sig | **4.1** |

**Veredicto: la config actual (A) ya es el sweet spot.**
- **top-10 no suma** (26.35 ≈ 26.46, ruido) pese a que el oracle-10 (19.3) es mejor que el oracle-5 (21.7):
  qwen no aprovecha las candidatas extra.
- **Los scores del beam no ayudan** (26.81) — el LLM no usa bien esa señal numérica.
- **El 9b es PEOR y 3.4× más lento** (27.52, pierde la significancia). Sorpresa: más parámetros ≠ mejor
  rescoring; el 4b-instruct está mejor calibrado para esta tarea acotada.
- La brecha al oracle (26.5 vs 21.7) **no se cierra con estos knobs** — quedaría para un rescorer entrenado
  (fine-tune del LLM sobre pares candidatas→referencia) o un LM acústico-consciente. No prioritario.

### ¿Cuántas candidatas / qué beam para el rescoring? (oracle-N vs beam, sin LLM)

Probe determinístico del **techo** (oracle = best-of-N por clip, elección perfecta), para decidir beam y
profundidad N *antes* de gastar en un sweep con LLM:

| beam | oracle N=1 | N=3 | N=5 | N=10 |
|---|---|---|---|---|
| 5 | 23.6 | 19.7 | 17.4 | 17.4 |
| 10 | 24.2 | 19.4 | 17.1 | 16.0 |
| 40 | 23.6 | 19.1 | 16.6 | 15.7 |

(IC ±6 en todas.) **Conclusiones:**
1. **El salto grande es N=1→N=3** (−3.9 de techo). Con **beam=3 el LLM ya tiene la mayor parte de la señal**;
   el rescoring funciona desde beam≥2 (solo beam=1/greedy no sirve, 1 sola candidata).
2. De 3→5 candidatas se gana ~2 más de *techo*, pero es oracle (qwen realiza solo parte) y a n=40 cae en el ruido.
3. **Beam más ancho (10/20/40) casi no mejora el oracle a N fijo** → no hay premio en beams anchos para el LLM;
   lo que importa es N, no el ancho. → El sweep completo "beam ancho + LLM" **no vale**; beam 3–5 es el rango.

→ **Demo:** `VSR_QWEN=1` usa beam 5 por default (mejor techo, +0.2 s vs beam3); podés bajar con `VSR_BEAM=3`
para el mínimo de latencia con corrección (~2.5 s/frase, 3 candidatas).

### ¿El LLM recupera la degradación de la cuantización int8? (beam 40, candidatas diversas)

Hipótesis: si int8 degrada el 1-best pero **preserva el pool** de candidatas, el rescoring podría recuperar
la pérdida (sería error de ranking, no de recall). Test con candidatas diversas (beam 40, ~4.8/5 únicas):

| beam40 | 1-best | oracle-5 | qwen-rescored |
|---|---|---|---|
| fp32 | 23.60 ± 6.6 | 16.57 | **20.22 ± 6.8** (−3.4) |
| int8 | 25.00 ± 7.2 | 17.42 | 23.60 ± 7.2 (−1.4) |
| brecha int8 | **+1.40** | +0.84 | **+3.37** |

- **No, el LLM NO recupera int8** — direccionalmente lo empeora: qwen ayudó a fp32 más (−3.4) que a int8
  (−1.4), así que la brecha creció de +1.4 (1-best) a +3.4 (tras qwen). Las candidatas limpias (fp32) son
  más fáciles de rescorear; el pool int8 se preserva casi igual (oracle +0.84) pero rinde menos.
- ⚠️ A n=40 esto caía dentro del ruido. **Repetido a n=100 con bootstrap pareado** (self-test ampliado, ver
  [05](05_selftest_limpio.md)): int8 tras qwen queda **−1.6 (IC cruza 0)** vs fp32 **−3.0 (IC [+0.71,+5.53],
  significativo)**; int8 vs fp32 tras qwen = +1.87, IC [0.00, 3.90] → **int8 sigue igual o peor, el LLM NO lo
  recupera** (ahora al borde de significancia, ya no puro ruido). El beneficio de qwen sí se volvió
  significativo con n=100; la recuperación de int8, no.
- **Retractación honesta:** en un test previo el rescoring de beam5 dio solo −0.3 (vs −3.4 con beam40) y lo
  atribuí a menor *diversidad* de candidatas. **Falso:** medido, beam5 y beam40 tienen la misma diversidad
  (4.80 vs 4.85 únicas/5). Esa diferencia de −0.3 vs −3.4 era **ruido de n=40**, no un efecto de diversidad.
- En la práctica no cambia nada en M1 (int8 ya está dominado por beam3). El principio "LLM como red de
  seguridad para compresión con pérdida" queda **sin probar a nuestro n**; revisitar para int8 server-side
  (x86/fbgemm, 2-3×) recién con el dataset ampliado.

## H. Recomendaciones (ordenadas por ROI, avaladas por el barrido §D)

1. **Bajar el beam de ViSpeR a 3 YA.** Es el punto de la frontera de Pareto: **mismo WER que beam40 (dentro
   del IC) a 2.2× la velocidad**, cero re-entrenamiento. Es cambiar un número
   (`get_beam_search_decoder(..., beam_size=3)` en `lightning_vsr.py`). **Aplicar en la demo y el server.**
2. **beam=1 (greedy) si se necesita más velocidad para streaming:** RTF 0.27 (3.8×) por +2.2 WER (borde del
   IC). Buen default para la ventana deslizante. **NO combinar con int8** (speedup nulo) ni subir ctc_weight.
3. **NO usar int8 en M1 ni subir ctc_weight:** el barrido mostró que **ambos están Pareto-dominados** —
   int8 agrega WER sin dar velocidad real (beam3 fp32 les gana), y ctc_weight alto es más lento *y* peor.
   (int8 solo reconsiderar en deploy x86/fbgemm.)
4. ~~El frontend es el muro~~ → **CAYÓ con el híbrido MPS** (§D): encoder en MPS 3.4× (0.57→0.17 s),
   transcripciones idénticas, ya cableado en `infer_server.py`. El cuello volvió a ser el **beam en CPU**
   (~0.9 s). Para bajar de ahí: beam1+MPS (~0.43 s) o CTC-greedy+MPS (~0.22 s, pero +12 WER). El siguiente
   salto real sigue siendo **destilar/achicar la arquitectura** (student causal, proyecto con GPU).
5. **Streaming (demo_stream.py):** implementado con **VAD visual** (corta en pausas de labios, auto-calibrado)
   + **transcript acumulado** + landmarks precomputados en vivo. Con encoder-MPS + beam3 la inferencia por
   segmento es ~1.1 s. Ver `realtime-vsr-plan` para la versión causal de verdad.
6. **Corrector qwen: la config actual es el sweet spot** (top-5, 4b, prompt plano; −3.04 SIG). top-10, scores
   y el 9b NO mejoran (§G A/B). +1.2 s/clip; para modo "calidad", no para el modo "más rápido posible".

**Conclusión:** aceleraciones reales encontradas: **beam 3** (2.2×, gratis) + **encoder en MPS** (frontend
3.4×, gratis y sin riesgo — 100/100 idéntico). int8, ctc_weight alto, CTC-greedy, top-10/scores/9b del LLM:
todos dominados o sin ganancia. Config de despliegue en M1: **encoder-MPS + beam3 (+ qwen top-5/4b opcional)**
→ ~1.1 s/clip solo, ~2.3 s con LLM, ambos bien bajo tiempo real.

---

**Fuentes (aceleración):** INT8 2.6–3.6× en CPU sin perder WER —
[4-bit LSTM ASR (arXiv:2108.12074)](https://arxiv.org/pdf/2108.12074),
[GPU-WFST beam decoder (arXiv:2311.04996)](https://arxiv.org/pdf/2311.04996) (el beam es ~90% del runtime CPU);
beam vectorizado — [arXiv:1811.04568](https://arxiv.org/pdf/1811.04568);
encoder eficiente — [SwinLip (arXiv:2505.04394)](https://arxiv.org/pdf/2505.04394);
[Uconv-Conformer (arXiv:2208.07657)](https://arxiv.org/pdf/2208.07657) (reduce la longitud de secuencia).
