# llm_corrector/ — corrección y rescoring con LLM

Componente de investigación LLM×VSR. La pregunta, el mapa completo de evidencia y las
limitaciones están en [`docs/RESEARCH.md`](../docs/RESEARCH.md); los números, en
[`../docs/experiments/04`](../docs/experiments/04_qwen_corrector.md) y el ledger.

El experimento definitivo del umbral —**grilla estandarizada de 13 celdas** (código, datos
crudos por clip, tabla máquina y figura)— vive en [`grid_cer_llm/`](grid_cer_llm/); los
resultados están escritos en [exp. 04 §G](../docs/experiments/04_qwen_corrector.md).

## Dónde vive cada estrategia (implementación canónica)

| Estrategia | Implementación canónica | Estado |
|---|---|---|
| Corrección 1-best + prompts (conservative/minimal/fluent) + few-shot | [`fase0_llm_correct.py`](fase0_llm_correct.py) (acá) | ❌ empeoró en todas las condiciones evaluadas (exp. 04 §A-D/F) |
| **n-best rescoring top-5** | prod: [`demo/infer_server.py`](../demo/infer_server.py) (prompt `SYS_RESC`, `VSR_QWEN=1`); research: [`grid_cer_llm/`](grid_cer_llm/) | ✅ ayuda en CER bajo + material limpio (pool −2.81 WER, IC [+0.86,+4.77]); nulo en YouTube — exp. 04 §F2/§G |
| Oracle n-best (techo) | scoring en [`personalization/score_selftest.py`](../personalization/score_selftest.py) + exp. 04 §E/F | referencia |
| Evaluación sobre self-test | [`personalization/score_selftest.py`](../personalization/score_selftest.py) | activa |

**Por qué el rescoring vive en `demo/infer_server.py` y no acá:** corre dentro del
proceso de inferencia (necesita las N candidatas del beam en memoria y el fallback
silencioso a 1-best); extraerlo a un módulo aparte agregaría un import cross-env
imposible de smoke-testear sin la máquina con el env `visper`. Este README es el índice
canónico; el código está enlazado.

## Requisitos

Ollama local con `qwen3:4b-instruct-2507-q4_K_M` (la variante "thinking" NO sirve —
razona en el output). Config: `VSR_QWEN`, `VSR_QMODEL`, `VSR_BEAM` (ver
[`.env.example`](../.env.example) y [SPEC §3.1](../docs/SPEC.md)).

## Resultados negativos (se preservan a propósito)

La corrección 1-best empeoró en los tres regímenes evaluados (CER 42/27/11); `fluent`
fue la variante más agresiva, few-shot quedó en nivel de ruido y el LLM también dañó texto
perfecto en el test de control. Estos resultados negativos son parte de la evidencia y se
preservan. Detalle: [exp. 04](../docs/experiments/04_qwen_corrector.md).
