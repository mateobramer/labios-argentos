# 04 — Corrector / rescoring con LLM (qwen)

**Hipótesis del usuario:** un modelo con **CER bajo** se corrige mejor con un LLM que uno con WER bajo
(los errores serían "typos" arreglables). Se probó exhaustivamente.

**Setup:** corrector = `qwen3:4b-instruct-2507-q4_K_M` (Ollama local, `think=false`, temp 0). Ojo: la
variante base "thinking" NO sirve (razona en el output y rompe todo). También disponible `qwen3.5:9b`.
Scripts: `llm_corrector/fase0_llm_correct.py`, y en scratchpad `qwen_lab.py`, `nbest_ft05.py`,
`visper_full_test.py`. Métrica: misma `norm()` + WER/CER, IC bootstrap.

---

## A. Corrección 1-best (Fase 0) — sobre 658 clips

| Modelo | CER base | %WER base→corr | %CER base→corr |
|---|---|---|---|
| ft05b | 42 | 70.30 → 71.04 (**+0.74 ❌**) | 42.07 → 43.77 (+1.70 ❌) |
| ft07 | 42 | 69.15 → 70.11 (**+0.96 ❌**) | 41.65 → 43.26 (+1.61 ❌) |
| **ViSpeR zero-shot** | 27 | 45.22 → 46.64 (**+1.42 ❌**) | 26.98 → 28.32 (+1.35 ❌) |

Estratificado por CER-por-clip: **empeora en TODOS los estratos**, y en ViSpeR el bucket más limpio
(CER 0-20) es el que MÁS empeora (+2.26). Motivo: los errores de lip-reading son **sustituciones por
otra palabra válida** (viseme-confusion), no typos → el LLM ciego al video lee español fluido, no tiene
señal de que está mal, y sobre-corrige lo que estaba bien.

**Conclusión A:** la corrección 1-best naive NO ayuda a CER 42 ni 27.

---

## B. Comparación de prompts — ft05b, 24 clips (base WER 69.13)

| Prompt | Δ WER | Δ CER | comportamiento |
|---|---|---|---|
| conservative | +0.26 | +0.83 | corrige de más |
| minimal | +0.00 | +0.67 | el más seguro, no mejora |
| fluent | +0.26 | +1.76 | **inventa** frases nuevas |

**Conclusión B:** ningún prompt rescata la corrección 1-best; "fluent" es el peor (alucina).

## C. qwen sobre el texto REAL (test de corrupción) — ft05b, 24 clips

Alimentar al LLM el texto CORRECTO (ground-truth): corrompe **2.04% WER / 0.93% CER** de texto ya perfecto
(ej: "me diga"→"me dijera"). → El corrector tiene un **piso de daño** aunque le des la verdad.

## D. Few-shot supervisado — ft05b, 24 clips, 5 demos hyp→ref (corrección ideal)

WER 69.13 → 68.88 (**−0.26**), CER +0.31. Nivel-ruido. Ni con demos de corrección ideal mejora.

---

## E. n-best rescoring — ft05 (50M), 40 clips (test-658)

En vez de corregir 1 hipótesis, darle al LLM las **top-N del beam** para que elija/combine.

| Enfoque | %WER |
|---|---|
| 1-best | 62.79 |
| **oracle-5** (mejor de 5, techo tramposo) | 55.04 |
| qwen n-best rescored | 62.64 (**−0.16**, no mejora) |

Las 5 candidatas del beam son **variaciones del mismo error** (la palabra correcta no está en ninguna)
→ ni el oracle baja de 55, y qwen no supera al 1-best. **A CER 42, el n-best tampoco alcanza.**

---

## F. ⭐ n-best rescoring — ViSpeR sobre self-test LIMPIO, 40 clips (CER ~11)

Acá el modelo base es bueno (CER 11) → el beam SÍ contiene la palabra correcta en candidatas bajas.

| Enfoque | %WER | %CER |
|---|---|---|
| 1-best (base) | 23.60 ± 6.6 | 11.41 ± 3.5 |
| qwen **1-best corregido** | 25.00 (**+1.40 ❌**) | 12.98 (+1.6 ❌) |
| **qwen n-best rescoring** | **20.22 (−3.37 ✅)** | **9.89 (−1.52 ✅)** |
| oracle-5 (techo) | 16.57 (−7.0) | 8.32 |

Ejemplos donde el n-best recupera lo correcto: "un hasta dos domingos"→"un asado el domingo";
"autocusado"→"auto usado"; "no te nos ganas"→"no tengo ganas".

### F2. ⭐⭐ Confirmación a n=100 (self-test ampliado) — AHORA SIGNIFICATIVO

Ampliamos el self-test a 100 clips (los 60 nuevos son más difíciles → WER base sube a 29.5, ver [05](05_selftest_limpio.md)).
Con **bootstrap PAREADO** del delta 1-best vs rescored:

| fp32 beam40 (n=100) | %WER | IC |
|---|---|---|
| 1-best | 29.51 | ±4.4 |
| **qwen n-best rescoring** | **26.46** | ±4.5 |
| oracle-5 (techo) | 21.66 | |
| **delta (1best−rescored)** | **−3.04** | **IC95 pareado [+0.71, +5.53] → excluye 0 ✅ SIGNIFICATIVO** |

El efecto (~3 WER) se mantuvo de n=40 (−3.4) a n=100 (−3.04) y **el IC pareado ya excluye el 0**. → **El n-best
rescoring baja el WER de verdad, no es ruido.** (Test int8: qwen le ayuda menos, −1.6 y cruza 0; el LLM no
recupera la degradación de int8. Ver [09](09_velocidad_inferencia.md) §G.)

---

## Conclusión general (hipótesis del CER)

- **La corrección 1-best NO ayuda a NINGÚN CER** probado (42, 27, 11) — sobre-corrige (over-correction).
- **El n-best rescoring SÍ ayuda cuando el CER es bajo (~11):** −3.4 WER / −1.5 CER. A CER alto (42) no,
  porque el beam no contiene la respuesta.
- **Tu hipótesis se confirma en dirección**, pero la llave es **n-best rescoring, no corrección 1-best**:
  el n-best le da *señal* (varias candidatas) que la corrección de una sola no tiene.
- **Cruce de signo del Δ WER:** CER 42 (+0.7) → 27 (+1.4) → 11 (−3.4 vía n-best). El umbral útil ≈ CER 20.
- ✅ **Significativo a n=100** (§F2): delta −3.04, IC95 pareado [+0.71, +5.53] excluye 0. (A n=40 era prometedor
  pero no concluyente; los 100 clips lo confirmaron.)
- Ojo: con 12 clips la corrección 1-best había dado −1.7 (falso positivo por ruido); el set de 40 lo corrigió.

**Pulido probado (n=100, ver [09](09_velocidad_inferencia.md) §G):** top-10 no suma (26.35≈26.46), scores en
el prompt no ayudan (26.81), y **qwen3.5:9b es PEOR y 3.4× más lento** (27.52, pierde significancia). → La
config actual (top-5, 4b-instruct, prompt plano) **ya es el sweet spot**. La brecha al oracle solo se
cerraría con un rescorer entrenado (no prioritario).
