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

Importante: esos casos son smoke tests derivados por reglas. Sirven para validar que
el codigo corre y que el provider respeta casos borde, pero no son ground truth
oracional real.

## Evaluacion causal por secuencia

El cierre real no corre por clip aislado: acumula clips de una misma fuente hasta
decidir si una oracion termino. Para eso existe:

```text
realtime/GROUND_TRUTH_ORACIONAL.md
realtime/src/secuencias.py
realtime/examples/ground_truth_demo.json
realtime/ground_truth/charla_amor_desamor.json
```

Exportar clips ordenados para anotarlos con un LLM potente o revision humana:

```bash
python -m realtime.src.secuencias export-annotation \
  --split vsr_models/splits/val.csv \
  --source-id "NOMBRE_EXACTO_DE_LA_FUENTE" \
  --limit 40 \
  --output realtime/outputs/annotation/fuente.md
```

Evaluar una secuencia anotada:

```bash
python -m realtime.src.secuencias evaluate \
  --ground-truth realtime/ground_truth/charla_amor_desamor.json
```

Metricas principales:

- commits tempranos (`early_commits`);
- waits tardios (`late_waits`);
- commits faltantes (`missing_commits`);
- precision/recall de commit;
- latencia p50/p95.

La primera fuente anotada es `CHARLA SOBRE EL AMOR Y EL DESAMOR`: 233 clips y
167 oraciones. Esa evaluacion ya muestra errores reales de la heuristica, y por
eso sirve como benchmark para comparar contra un LLM zero-shot o un futuro
student causal entrenado.

## Entrenamiento liviano de cierre

El cierre en vivo conviene tratarlo como un clasificador de baja latencia, separado
del corrector. El flujo esperado es:

```text
VSR por chunks -> buffer textual -> cierre liviano -> oracion commiteada -> corrector
```

El entrenamiento compara varios modos sin dependencias externas:

- `majority`: baseline de clase mayoritaria.
- `heuristic`: baseline interpretable actual.
- `linear_text`: perceptron multiclase con features textuales.
- `linear_text_balanced`: igual, con balanceo por clase.
- `linear_heuristic`: features textuales + decision/razon de la heuristica.
- `linear_heuristic_balanced`: version balanceada.

Entrenar y seleccionar el mejor:

```bash
python -m realtime.src.entrenar_cierre \
  --input realtime/ground_truth \
  --input realtime/outputs/synthetic_cierre_oracional \
  --output-dir realtime/outputs/cierre_training
```

El output principal es:

```text
realtime/outputs/cierre_training/summary.json
realtime/outputs/cierre_training/best_config.json
realtime/outputs/cierre_training/*.model.json
```

La seleccion usa `val.selection_score`, que prioriza `commit_f1` y penaliza fuerte los
commits prematuros. En esta tarea un `commit` temprano corta una idea y suele ser peor
que esperar un clip mas.

Usar un modelo entrenado en el simulador:

```bash
python -m realtime.src.simular_flujo \
  --demo \
  --closure-provider linear \
  --model-path realtime/outputs/cierre_training/linear_heuristic.model.json
```

Evaluar una secuencia anotada con el modelo entrenado:

```bash
python -m realtime.src.secuencias evaluate \
  --ground-truth realtime/ground_truth/charla_amor_desamor.json \
  --provider linear \
  --model-path realtime/outputs/cierre_training/linear_heuristic.model.json
```

## Data sintetica con GPT Pro

Prompt y guia:

```text
realtime/GPT_PRO_SYNTHETIC_PROMPT.md
```

Generar un plan de variaciones por lotes:

```bash
python -m realtime.src.plan_sintetico \
  --shuffle \
  --seed 13 \
  --max 80 \
  --output realtime/outputs/synthetic_plan/lote_001.jsonl
```

La grilla completa de factores tiene 5040 variaciones. Conviene empezar con lotes de
80-200, revisar calidad y escalar. Mantener `synthetic=true` y no mezclar esos datos
ciegamente con ground truth real.

## Notebooks

Los notebooks son livianos, estan ejecutados con outputs guardados y estan pensados
para que alguien que entra al repo entienda que se hizo sin correr GPU ni servicios
externos:

- `notebooks/01_cierre_heuristico.ipynb`: contratos, cierre heuristico y metricas demo.
- `notebooks/02_simulador_feedback_logs.ipynb`: flujo completo, feedback JSONL y logs.
- `notebooks/03_casos_desde_splits.ipynb`: ground truth oracional, export para LLM
  fuerte y evaluacion causal por secuencia.

## Tests

```bash
python -m unittest discover -s realtime/tests
```

## Etapas siguientes

1. Probar y medir el provider Ollama/Qwen con salida JSON estricta.
2. Revisar/ampliar el ground truth de `CHARLA SOBRE EL AMOR Y EL DESAMOR`.
3. Evaluar cierre causal sobre esa fuente y comparar heuristica vs LLM local/API.
4. Evaluar offline contra `evaluation/outputs/*/test.inf`.
5. Integrar el corrector real de Mateo detras de `CorrectionProvider`.
6. Convertir feedback validado en dataset revisable para evaluacion o fine-tuning.
