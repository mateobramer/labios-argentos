# realtime

Modulo para el flujo en tiempo real del proyecto: cierre de oracion, contrato con
corrector LM y feedback auditable.

La etapa inicial es deliberadamente conservadora:

- no requiere GPU;
- no requiere Ollama;
- no requiere API externa;
- no agrega dependencias nuevas;
- ante fallas, espera (`wait`) o conserva texto crudo.

## Flujo

```text
texto parcial del VSR
-> cierre de oracion
-> texto commiteado
-> corrector
-> texto corregido
-> feedback JSONL
-> evaluacion / revision / futuros fine-tunes
```

## Contratos

Los contratos viven en `realtime/src/contracts.py`.

Decision de cierre:

```json
{
  "action": "commit | wait | low_confidence",
  "committed_text": "",
  "confidence": 0.0,
  "reason": "",
  "risk_flags": []
}
```

Reglas duras:

- si `action` no es `commit`, `committed_text` queda vacio;
- si la salida de un provider es invalida, el cierre cae a `wait`;
- si el corrector falla, se conserva el texto crudo;
- ningun provider debe romper el loop principal con una excepcion sin capturar.

## Providers actuales

- `HeuristicClosureProvider`: cierre conservador sin dependencias.
- `IdentityCorrectionProvider`: corrector no-op que devuelve el texto crudo.
- `OllamaProvider`: provider opcional para cierre/correccion via Ollama, con JSON schema
  y fallback seguro. No se usa por defecto.

Los providers futuros de Ollama, OpenAI, llama.cpp o PyTorch deben implementar las
interfaces de `realtime/src/llm/providers.py` sin volver obligatorias esas dependencias.

## Feedback y logs

Los outputs locales se escriben en:

```text
realtime/outputs/feedback/
realtime/outputs/llm_logs/
```

Esos outputs son para debugging/evaluacion local y no deben versionarse.

Cada evento de feedback queda como JSONL revisable. Una correccion de usuario no entra
automaticamente al entrenamiento: primero queda con `review_status=pending`.

## Simulador offline

Demo sin GPU ni LLM externo:

```bash
python -m realtime.src.simular_flujo --demo
```

Probar el mismo flujo con cierre via Ollama, solo si Ollama ya esta levantado:

```bash
python -m realtime.src.simular_flujo --demo --closure-provider ollama --ollama-model qwen3:4b
```

Con archivo propio, una hipotesis parcial por linea:

```bash
python -m realtime.src.simular_flujo --input ejemplos.txt
```

El simulador reporta:

- cantidad de ejemplos;
- commits / waits / low_confidence;
- latencia p50/p95 de cierre;
- latencia p50/p95 de correccion;
- latencia p50/p95 de validacion;
- latencia p50/p95 de logging;
- cantidad de fallbacks.

## Evaluacion offline de cierre

Casos sinteticos etiquetados:

```bash
python -m realtime.src.evaluar_cierre --demo
```

Con dataset JSONL:

```bash
python -m realtime.src.evaluar_cierre --input cierre_eval.jsonl
```

Formato esperado por linea:

```json
{"partial_text": "yo creo que", "expected_action": "wait"}
```

Metricas reportadas:

- accuracy;
- precision/recall de `commit`;
- tasa de commit prematuro;
- tasa de wait innecesario;
- recall de `low_confidence`;
- latencia p50/p95;
- fallbacks.

Tambien se pueden generar casos livianos desde los splits del proyecto, sin cargar ROIs:

```bash
python -m realtime.src.dataset_cierre \
  --split vsr_models/splits/val.csv \
  --limit 30 \
  --output realtime/outputs/eval/cierre_val.jsonl
```

Luego se evaluan con:

```bash
python -m realtime.src.evaluar_cierre --input realtime/outputs/eval/cierre_val.jsonl
```

## Notebooks

Los notebooks son livianos, estan ejecutados con outputs guardados y estan pensados
para que alguien que entra al repo entienda que se hizo sin correr GPU ni servicios
externos:

- `notebooks/01_cierre_heuristico.ipynb`: contratos, cierre heuristico y metricas demo.
- `notebooks/02_simulador_feedback_logs.ipynb`: flujo completo, feedback JSONL y logs.
- `notebooks/03_casos_desde_splits.ipynb`: casos de evaluacion desde `vsr_models/splits/*.csv`.

## Tests

```bash
python -m unittest discover -s realtime/tests
```

## Etapas siguientes

1. Probar y medir el provider Ollama/Qwen con salida JSON estricta.
2. Evaluar offline contra `evaluation/outputs/*/test.inf`.
3. Comparar heuristica vs LLM local/API.
4. Integrar el corrector real de Mateo detras de `CorrectionProvider`.
5. Convertir feedback validado en dataset revisable para evaluacion o fine-tuning.
