# Evaluacion del corrector LLM

Esta etapa se corre despues de tener predicciones VSR de `baseline_original` y
`visual_cleaned`. No entrena VSR y no modifica labels.

## Input esperado

CSV por experimento con schema VSR:

```csv
experiment,source_id,clip,split,training_usability,policy_moderate,reference,hypothesis,wer,cer
```

El corrector recibe `hypothesis` y puede usar contexto minimo del experimento. La
referencia se usa solo para medir despues, no para generar la correccion.

## Protocolo de prompt

Instrucciones minimas:

- Corregir solo errores ortograficos o gramaticales plausibles de la salida VSR.
- Mantener el contenido semantico observado.
- No inventar nombres propios, numeros, entidades, fechas ni palabras ausentes.
- No borrar disfluencias si cambian el contenido hablado.
- Devolver solo texto corregido.

## Output esperado

```csv
experiment,source_id,clip,split,reference,raw_hypothesis,corrected_hypothesis,raw_wer,corrected_wer,raw_cer,corrected_cer,llm_changed,llm_improved,llm_worsened,notes
```

## Metricas

- Raw WER/CER vs corrected WER/CER.
- `improved`, `worsened`, `unchanged`.
- Tasa de edicion: porcentaje de clips con `llm_changed == true`.
- Casos donde baja CER pero sube WER.
- Casos donde el corrector borra palabras.
- Casos donde inventa entidades o completa contenido no observado.

## Guardrails

- El LLM no modifica transcripciones de entrenamiento.
- El LLM no se usa para seleccionar clips.
- El LLM no ve `reference` durante la generacion.
- Si el corrector mejora metricas pero aumenta alucinaciones, no se reporta como mejora
  final sin inspeccion manual.

## Estado

Esta línea exploratoria (evaluar el corrector LLM sobre esta muestra específica de
visual_cleaning) no llegó a tener un runner que leyera el CSV VSR y aplicara el corrector.
La evaluación real del corrector LLM se hizo por otra vía: ver
[`docs/experiments/04_qwen_corrector.md`](../../../../docs/experiments/04_qwen_corrector.md).
