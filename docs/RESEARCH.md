# Research — LLMs × VSR: qué ayuda, qué perjudica, y bajo qué condiciones

**Pregunta de investigación:** ¿qué estrategias de uso de LLMs ayudan o perjudican al
reconocimiento visual del habla (VSR) en español rioplatense, y bajo qué condiciones?

Este doc es el **índice de la evidencia** — los números viven en
[`experiments/04_qwen_corrector.md`](experiments/04_qwen_corrector.md) (doc primario)
y el ledger [`RESULTS.md`](RESULTS.md). No se ocultan resultados negativos: son la
mayor parte del hallazgo.

## Mapa de la evidencia

| # | Estrategia | Condición | Resultado | Dónde |
|---|---|---|---|---|
| 1 | Corrección directa 1-best | CER base 42 (ft05b/ft07) | ❌ empeora (+0.7/+1.0 WER) | [04 §A](experiments/04_qwen_corrector.md) |
| 2 | Corrección directa 1-best | CER base 27 (ViSpeR zs, 658 clips) | ❌ empeora (+1.42 WER) | [04 §A](experiments/04_qwen_corrector.md) |
| 3 | Corrección directa 1-best | CER base ~11 (self-test limpio) | ❌ empeora (+1.40 WER) | [04 §F](experiments/04_qwen_corrector.md) |
| 4 | Prompts (conservative / minimal / fluent) | CER 42, 24 clips | ❌ ninguno rescata; "fluent" alucina | [04 §B](experiments/04_qwen_corrector.md) |
| 5 | LLM sobre texto ground-truth (test de corrupción) | texto perfecto | ❌ corrompe 2.04% WER — piso de daño | [04 §C](experiments/04_qwen_corrector.md) |
| 6 | Few-shot supervisado (5 demos hyp→ref) | CER 42, 24 clips | ❌ nivel-ruido (−0.26 WER) | [04 §D](experiments/04_qwen_corrector.md) |
| 7 | n-best rescoring (top-5) | CER base 42 (ft05) | ❌ no mejora (−0.16); ni el oracle baja de 55 | [04 §E](experiments/04_qwen_corrector.md) |
| 8 | **n-best rescoring (top-5)** | **CER base ~11, n=40** | ✅ −3.37 WER / −1.52 CER | [04 §F](experiments/04_qwen_corrector.md) |
| 9 | **n-best rescoring — confirmación n=100** | CER ~11, bootstrap pareado | ✅ **−3.04 WER, IC95 [+0.71, +5.53] excluye 0 → significativo** | [04 §F2](experiments/04_qwen_corrector.md) |
| 10 | Oracle n-best (techo) | CER ~11, n=100 | techo −7.85 WER (21.66) — brecha restante | [04 §F2](experiments/04_qwen_corrector.md) |
| 11 | Pulido: top-10, scores en prompt, qwen 9b | CER ~11, n=100 | ❌ ninguno mejora; 9b es peor y 3.4× más lento | [04 conclusión](experiments/04_qwen_corrector.md), [09 §G](experiments/09_velocidad_inferencia.md) |
| 12 | n-best sobre modelo degradado (int8) | int8/greedy | ❌ el LLM no recupera la degradación (−1.6, cruza 0) | [09 §G](experiments/09_velocidad_inferencia.md) |
| 13 | LLM para limpieza de transcripciones del dataset | offline, GPT | revisión en notebook; parches aceptados/rechazados registrados | [cleaning/visual_quality nb 03](../cleaning/visual_quality/notebooks/03_revision_correcciones_llm.ipynb), [nb 08](../cleaning/visual_quality/notebooks/08_transcript_cleaning_review.ipynb), `cleaning/gpt_clean_v1/` |
| 14 | Redundancia LoRA personal ↔ qwen | hablante calibrado | los beneficios se solapan parcialmente | [10](experiments/10_adaptacion_hablante.md) |

## Los hallazgos centrales

1. **La corrección 1-best empeoró en todas las condiciones evaluadas (CER 42, 27 y 11).** El motivo
   está diagnosticado: los errores de lip-reading son sustituciones por palabras válidas
   (confusión de visemas), no typos. Un LLM ciego al video lee español fluido y
   sobre-corrige lo que estaba bien. El test de corrupción (#5) muestra que el corrector
   tiene un piso de daño incluso con entrada perfecta.
2. **La evidencia sugiere que el LLM solo puede ayudar cuando la respuesta correcta está en el beam.**
   A CER 42 el beam contiene variaciones del mismo error (oracle apenas 55) y el LLM no
   tiene de dónde elegir. A CER ~11 el beam sí contiene la palabra correcta en
   candidatas bajas, y el rescoring la recupera ("autocusado"→"auto usado").
3. **El cruce de signo del Δ WER** con el CER base: +0.7 (CER 42) → +1.4 (CER 27) →
   −3.4 vía n-best (CER 11). Umbral útil estimado ≈ CER 20.
4. **Significancia:** el efecto −3 WER se sostuvo de n=40 a n=100 y el IC95 pareado
   excluye 0. Con n=12 la corrección 1-best había dado un falso positivo (−1.7) que
   n=40 desmintió — lección metodológica sobre tamaños de muestra chicos.
5. **La mejor configuración entre las evaluadas fue:** top-5, qwen3:4b-instruct, prompt plano, temp 0.
   Escalar el LLM (9b) o el n (10) no mejora. La brecha al oracle (~5 WER) requeriría
   un rescorer entrenado.

## Cómo reproducir (hasta donde permiten los artefactos)

- Scripts: `llm_corrector/fase0_llm_correct.py` (corrección 1-best);
  el rescoring vive integrado en `demo/infer_server.py` (`VSR_QWEN=1`, prompt en
  `SYS_RESC`) y `personalization/score_selftest.py` evalúa sobre el self-test.
- Requiere: env `visper` + pesos ViSpeR (ver [`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md)),
  Ollama con `qwen3:4b-instruct-2507-q4_K_M`, y los clips del self-test (locales, no
  versionados por privacidad) o `test-658` (splits congelados en `vsr/splits/`).
- La misma `norm()` en todas las evaluaciones: minúsculas, sin acentos (ñ preservada),
  sin puntuación. Bootstrap 2000 iters para IC; **pareado** para deltas.

## Limitaciones estadísticas y de alcance (honestas)

- El resultado significativo (#9) es sobre **self-test de 1 hablante** en condiciones
  controladas (n=100). En test-658 (multi-hablante, YouTube) el n-best no se re-midió a
  n grande porque el CER base (27) está sobre el umbral útil estimado.
- Los experimentos de prompts/few-shot (#4, #6) usan n=24 — orientativos, no concluyentes.
- Un solo LLM local evaluado en profundidad (familia qwen3); no se probaron APIs
  comerciales (decisión de privacidad/costo, ver [SPEC §7](SPEC.md)).
- El umbral CER ≈ 20 es una interpolación de 3 puntos, no un barrido fino.

## Extensiones naturales (ver [FUTURE_WORK](FUTURE_WORK.md))

Rescorer entrenado (cerrar brecha al oracle), barrido fino del umbral de CER,
replicación multi-hablante del n-best, comparación qwen local vs API con
costo/latencia/privacidad.
