# Matriz batch VSR E0-E4

Esta etapa prepara configuracion, no entrena.

| Experimento | visual_cleaning | transcript_variant | preprocessing_variant | Estado esperado |
|---|---|---|---|---|
| E0_baseline_original | none | current | current | ready |
| E1_visual_cleaned | conservative | current | current | ready |
| E2_transcript_cleaned_stronger | none | transcript_cleaned_stronger | current | ready |
| E3_preprocessing_variant | none | current | lower_face_resized96 | ready_after_generation |
| E4_all_combined | conservative | transcript_cleaned_restricted | lower_face_resized96 | ready_after_generation |

`ready_after_generation` significa que la config esta definida, pero antes de entrenar
hay que generar `evaluation/outputs/batch_vsr/rois_lower_face_resized96/`.

## Configs

Generar con:

```bash
python -m evaluation.src.build_batch_vsr_experiments \
  --output-base evaluation/outputs/batch_vsr
```

Salidas:

```text
evaluation/outputs/batch_vsr/experiments/E0_baseline_original/experiment_config.json
evaluation/outputs/batch_vsr/experiments/E1_visual_cleaned/experiment_config.json
evaluation/outputs/batch_vsr/experiments/E2_transcript_cleaned/experiment_config.json
evaluation/outputs/batch_vsr/experiments/E3_preprocessing_variant/experiment_config.json
evaluation/outputs/batch_vsr/experiments/E4_all_combined/experiment_config.json
```

## Decisiones

- Test principal: siempre `evaluation/outputs/visual_cleaning/manifests/original_test.csv`.
- Splits canonicos: no se modifican.
- Transcripts originales: no se modifican.
- ROIs originales: no se modifican.
- `E2` usa `splits_transcript_cleaned_stronger/` y overlay
  `transcripts_cleaned_stronger/`.
- `E4` usa `splits_all_combined/`, `transcripts_cleaned_stronger/` y
  `lower_face_resized96`.
- `E3` y `E4` requieren generar ROIs alternativos antes de entrenamiento.
- `transcript_policy_moderate` excluye del train solo `bad_candidate`; mantiene
  `questionable`.
