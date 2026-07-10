# experiments/ — registro completo de experimentos VSR rioplatense

Índice de TODOS los experimentos del proyecto con sus resultados. Cada doc cubre una categoría.
Métricas: **%WER / %CER** con IC 95% (bootstrap) donde aplica. Salvo aclaración, todo es sobre
**`test-658`** = 658 clips, 2 hablantes held-out (f15/f22), speaker-independent, split congelado desde ft03.
Normalización idéntica en todo: minúsculas, sin acentos (ñ preservada), sin puntuación.

**Arquitectura común (modelos "propios", el 50M):** Conv3D+ResNet18 → Conformer(12) → decoder híbrido
CTC/Attention. Offline/bidireccional. ViSpeR = misma familia pero 288M (adim768).

## Docs

- [01 — Fine-tunes del 50M (Gimeno)](01_finetunes_50M_gimeno.md) — ft03…ft07.
- [02 — Zero-shots](02_zeroshot.md) — mpc001 CMU-MOSEAS, ViSpeR.
- [03 — Fine-tunes de ViSpeR](03_visper_finetunes.md) — full-FT y LoRA+aug.
- [04 — Corrector LLM (qwen)](04_qwen_corrector.md) — TODOS los experimentos con qwen.
- [05 — Self-test limpio (usuario)](05_selftest_limpio.md) — grabaciones controladas; hipótesis del CER.
- [06 — Demo tiempo real + remap ft05](06_demo_y_remap.md) — push-to-talk; ft05 espnet2→espnet1.
- [07 — Datos y scraping](07_datos_y_scraping.md) — corpus disponibles, dataset argentino, pared de YouTube.
- [08 — Benchmarks externos](08_benchmarks_externos.md) — Auto-AVSR, Gimeno, etc. (contexto).
- [09 — Velocidad de inferencia](09_velocidad_inferencia.md) — timing ViSpeR vs ft05, beam/int8 sweep, latencia del LLM, aceleración.
- [10 — Adaptación al hablante](10_adaptacion_hablante.md) — LoRA personal sobre ViSpeR (60 clips): −4.7 WER personal sin olvido; full-FT colapsa; redundancia con qwen.

## TABLA MAESTRA (sobre test-658, salvo aclaración)

| # | Modelo / enfoque | Datos train | %WER | %CER | Nota |
|---|---|---|---|---|---|
| 1 | mpc001 CMU-MOSEAS ES **zero-shot** | 0 (16h pretrain) | 71.50 | 46.88 | base multilingüe chica |
| 2 | ft03 (LIP-RTVE + AR) | 4818 (~9h) | 68.93 | 41.01 | ronda-1, v1 full-FT |
| 3 | ft04 (LIP-RTVE + AR) | ~4800 | 69.73 | 42.29 | ronda-1, v2 freeze+aug |
| 4 | **ft05** ⭐ (LIP-RTVE + AR) | 10934 (~19h) | **65.05** | 38.24 | ronda-2, v1 — mejor propio |
| 5 | ft06 (LIP-RTVE + AR) | ~10900 | 66.37 | 39.34 | ronda-2, v2 |
| 6 | ft05b (LIP-RTVE, same-data) | 8067 | 70.30 | 42.08 | gemelo reproducible de ft05 |
| 7 | ft07 (multiling. remap, same-data) | 8067 | 69.15 | 41.66 | A/B de bases |
| 8 | **ViSpeR zero-shot** ⭐⭐ | 0 (794h pretrain) | **45.22** | 26.98 | mejor / más simple |
| 9 | ViSpeR full-FT AR | 8067 | 61.51 | 38.59 | ❌ overfit (empeora) |
| 10 | ViSpeR LoRA+aug AR | 8067 | 45.97 | 26.00 | ≈ empata zero-shot |
| 11 | ViSpeR zs + corrector qwen | — | 46.64 | 28.32 | ❌ el LLM empeora |
| 12 | **ViSpeR + LoRA personal (Fede)** | 60 clips propios | **44.54** | 25.94 | sin olvido; personal 29.2→24.5 ([10](10_adaptacion_hablante.md)) |
| 13 | ViSpeR + full-FT personal lr1e-5 | 60 clips propios | 98.69 ☠️ | 90.57 | colapso — la receta del 50M no escala a 288M |

**Sobre self-test LIMPIO del usuario (NO test-658, condiciones ideales):** ver [05](05_selftest_limpio.md).
- 40 clips: ViSpeR 1-best **23.60 / 11.41** · qwen n-best **20.22 / 9.89** · ft05 ~68 WER.
- **100 clips** (60 nuevos, más difíciles): ViSpeR 1-best **29.51** · qwen n-best **26.46**, delta **−3.04
  con IC95 pareado [+0.71, +5.53] → SIGNIFICATIVO ✅** (el beneficio de qwen dejó de ser ruido).

## Conclusiones globales

1. **La escala de pre-entrenamiento domina, no la arquitectura.** ViSpeR (794h) zero-shot le gana por
   ~20 WER a nuestro mejor fine-tune propio (ft05, 65). Cambiar de base (ft05b vs ft07) o de config casi
   no mueve; **más datos** sí (ft03→ft05: −3.9 WER).
2. **El 50M NO está limitado por capacidad, sino por datos** (mpc001 50M igualó a Auto-AVSR 250M en LRS3
   con datos suficientes). ft05 no mejora ni en condiciones ideales → su techo es el pre-entreno chico.
3. **Fine-tunear con ~19h no supera al zero-shot de ViSpeR** (full-FT overfitea; LoRA+aug empata).
4. **El corrector LLM 1-best NO ayuda** a ningún CER (42, 27, 11). PERO **n-best rescoring SÍ ayuda a
   CER bajo**: **−3.04 WER, SIGNIFICATIVO a n=100** (IC95 pareado [+0.71, +5.53] excluye 0). La corrección
   1-best empeora; el n-best rescoring baja el WER de verdad. En int8/greedy no aplica (§09).
5. **No existe corpus AV rioplatense público**; el scraping masivo de YouTube no es viable (pared anti-bot).
6. **Velocidad ([09](09_velocidad_inferencia.md)):** dos aceleraciones reales y gratis: **beam 3** (2.2×,
   mismo WER) y **encoder en MPS** (frontend 3.4×: 0.57→0.17 s, transcripciones 100/100 idénticas, cableado
   en el demo). int8, ctc_weight alto, CTC-greedy: **Pareto-dominados**. El corrector qwen (top-5/4b, la
   config óptima — top-10/scores/9b no mejoran) agrega ~1.2 s por **−3.04 WER significativo**. Config M1:
   **encoder-MPS + beam3** ≈ 1.1 s/clip (~2.3 s con LLM), bajo tiempo real. Streaming con VAD visual +
   transcript acumulado: `demo/demo_stream.py`.
7. **Adaptación al hablante ([10](10_adaptacion_hablante.md)):** LoRA con 60 clips propios baja el WER
   personal 29.2→24.5 (−4.7, aún no sig. a n=30) **sin olvido** (test-658 mejora una pizca). full-FT
   lr1e-5 (receta del 50M del compañero) **colapsa el 288M** — con modelos grandes, PEFT o nada. Ojo:
   adaptación y rescoring qwen resultaron **redundantes** (explotan el mismo error recuperable).
