# multilingual-vsr/ — base multilingüe mpc001 (notas y scripts)

Trabajo sobre la base **CMU-MOSEAS ES multilingüe (mpc001)**: el clon del repo externo
(`multilingual-vsr/repo/`) y las corridas (`runs/`) **no se versionan** (.gitignore);
acá viven las notas y los scripts propios.

| Path | Qué es |
|---|---|
| `notes/01_hallazgos_repo.md` | hallazgos al examinar el repo externo |
| `notes/02_plan_realtime_y_benchmarks.md` | plan de tiempo real y benchmarks (histórico) |
| `scripts/zeroshot.py` | evaluación zero-shot — **define la `norm()` canónica** del proyecto (minúsculas, sin acentos, ñ preservada, sin puntuación) |
| `scripts/fase0_llm_correct.py` | corrección LLM 1-best (fase 0) — evidencia de [exp. 04 §A-D](../experiments/04_qwen_corrector.md) |

Resultados asociados: zero-shot mpc001 en [exp. 02](../experiments/02_zeroshot.md),
remap a espnet1 en [exp. 06](../experiments/06_demo_y_remap.md).
