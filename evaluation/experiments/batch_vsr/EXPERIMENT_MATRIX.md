# Matriz batch VSR E0-E4

Esta etapa prepara configuracion y readiness; el entrenamiento corre en VM.

| Experimento | visual_cleaning | transcript_variant | preprocessing_variant | Estado local |
|---|---|---|---|---|
| E0_baseline_original | none | current | current | ready |
| E1_visual_cleaned | conservative | current | current | ready |
| E2_transcript_cleaned_stronger | none | transcript_cleaned_stronger | current | blocked_missing_asr2 |
| E3_preprocessing_variant | none | current | lower_face_resized96 | ready_after_generation |
| E4_all_combined | conservative | current hasta que E2 este listo | lower_face_resized96 | ready_after_generation |

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
evaluation/outputs/batch_vsr/experiments/E2_transcript_cleaned_stronger/experiment_config.json
evaluation/outputs/batch_vsr/experiments/E3_preprocessing_variant/experiment_config.json
evaluation/outputs/batch_vsr/experiments/E4_all_combined/experiment_config.json
```

## Decisiones

- Test principal: siempre `evaluation/outputs/visual_cleaning/manifests/original_test.csv`.
- Splits canonicos, transcripts originales y ROIs originales no se modifican.
- `E2` se entrena solo si ASR2 real deja `transcript_decision == READY_FOR_VM`.
- Si E2 queda `BLOCKED_MISSING_ASR2` o `LOW_IMPACT_DO_NOT_PRIORITIZE`, `E4` corre como
  visual cleaning + `lower_face_resized96`, sin depender de transcript cleaned.
- `transcript_policy_moderate` excluye del train solo `bad_candidate`; mantiene
  `questionable`.
