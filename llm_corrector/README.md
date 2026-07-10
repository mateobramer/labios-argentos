# llm_corrector/ — corrección y rescoring con LLM

Componente de investigación LLM×VSR. La pregunta, el mapa completo de evidencia y las
limitaciones están en [`docs/RESEARCH_TP.md`](../docs/RESEARCH_TP.md); los números, en
[`../docs/experiments/04`](../docs/experiments/04_qwen_corrector.md) y el ledger.

## Dónde vive cada estrategia (implementación canónica)

| Estrategia | Implementación canónica | Estado |
|---|---|---|
| Corrección 1-best + prompts (conservative/minimal/fluent) + few-shot | [`fase0_llm_correct.py`](fase0_llm_correct.py) (acá) | ❌ negativa — nunca ayudó (exp. 04 §A-D) |
| **n-best rescoring top-5** (la que SÍ funciona) | [`demo/infer_server.py`](../demo/infer_server.py) — prompt `SYS_RESC`, activable con `VSR_QWEN=1` | ✅ −3.04 WER significativo a CER bajo (exp. 04 §F2) |
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

La corrección 1-best empeora a TODO CER probado (42/27/11); "fluent" alucina; few-shot
es ruido; el LLM corrompe hasta texto perfecto (piso de daño 2 % WER). Es la mitad del
hallazgo del TP de Research — no borrar. Detalle: [exp. 04](../docs/experiments/04_qwen_corrector.md).
