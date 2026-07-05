# Schema de resultados

Los CSV de resultados deben vivir bajo `evaluation/outputs/visual_cleaning/`.

## VSR

Archivo esperado por experimento:

```text
evaluation/outputs/visual_cleaning/results/baseline_original_test.csv
evaluation/outputs/visual_cleaning/results/visual_cleaned_test_original.csv
```

Columnas:

```csv
experiment,source_id,clip,split,training_usability,policy_moderate,reference,hypothesis,wer,cer
```

Definiciones:

- `experiment`: `baseline_original` o `visual_cleaned`.
- `source_id`: fuente/titulo normalizado usado en manifests.
- `clip`: id del clip, por ejemplo `clip_0001`.
- `split`: debe ser `test` para el numero principal.
- `training_usability`: etiqueta del manifest visual (`usable`, `questionable`,
  `bad_candidate`).
- `policy_moderate`: decision de politica visual usada para entrenamiento.
- `reference`: texto real del clip.
- `hypothesis`: salida cruda del VSR.
- `wer`: word error rate por clip.
- `cer`: character error rate por clip.

Reglas:

- El test principal no se filtra.
- No concluir por grupo si el grupo tiene menos de 30 clips.
- Si faltan `wer` o `cer`, calcularlos con `evaluation.src.experiment_metrics`.

## LLM corrector

Archivo esperado por experimento:

```text
evaluation/outputs/visual_cleaning/llm_corrector/baseline_original_llm_corrected.csv
evaluation/outputs/visual_cleaning/llm_corrector/visual_cleaned_llm_corrected.csv
```

Columnas:

```csv
experiment,source_id,clip,split,reference,raw_hypothesis,corrected_hypothesis,raw_wer,corrected_wer,raw_cer,corrected_cer,llm_changed,llm_improved,llm_worsened,notes
```

Definiciones:

- `raw_hypothesis`: salida VSR antes del corrector.
- `corrected_hypothesis`: salida post-procesada por el corrector.
- `llm_changed`: `true` si la salida cambio.
- `llm_improved`: `true` si bajo WER o CER sin marcar riesgo en `notes`.
- `llm_worsened`: `true` si subio WER o CER.
- `notes`: observaciones de alucinacion, borrado, entidad inventada o caso dudoso.

Regla: el corrector opera solo sobre predicciones, nunca sobre labels de entrenamiento.
