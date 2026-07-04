# VM readiness batch VSR

Fecha local: 2026-07-04.

## Experimentos

| Experimento | status | train_split | val_split | test_split | rois_root | transcripts_root |
|---|---|---|---|---|---|---|
| E0_baseline_original | READY | `evaluation/outputs/visual_cleaning/splits_original/train.csv` | `evaluation/outputs/visual_cleaning/splits_original/val.csv` | `evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `data/processed/lip_rois` | columna `texto` |
| E1_visual_cleaned | READY | `evaluation/outputs/visual_cleaning/splits_visual_cleaned/train.csv` | `evaluation/outputs/visual_cleaning/splits_visual_cleaned/val.csv` | `evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `data/processed/lip_rois` | columna `texto` |
| E2_transcript_audited | BLOCKED_MISSING_ASR2 | `evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/train.csv` | `evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/val.csv` | `evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `data/processed/lip_rois` | `evaluation/outputs/batch_vsr/transcripts_cleaned_stronger` |
| E3_preprocessing_variant | READY_AFTER_FULL_GENERATION | `evaluation/outputs/visual_cleaning/splits_original/train.csv` | `evaluation/outputs/visual_cleaning/splits_original/val.csv` | `evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `evaluation/outputs/batch_vsr/rois_lower_face_resized96` | columna `texto` |
| E4_all_combined | READY_AFTER_FULL_GENERATION | `evaluation/outputs/visual_cleaning/splits_visual_cleaned/train.csv` | `evaluation/outputs/visual_cleaning/splits_visual_cleaned/val.csv` | `evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `evaluation/outputs/batch_vsr/rois_lower_face_resized96` | columna `texto` hasta que E2 quede READY_FOR_VM |

## Transcript status

```text
auto_clean changes: 52
ASR2 status: blocked_missing_asr_dependency local smoke, 20 rows
disagreement rows: 20 blocked
replacement candidates: 48
auto replacements: 0
questionable: 48
bad_candidate: 0
excluded_train: 0
decision: BLOCKED_MISSING_ASR2
```

`BLOCKED_MISSING_ASR2`: text-only audit cannot detect hallucinations/misalignment reliably.
No correr E2 como mejora fuerte hasta generar ASR2 real en VM.

## Preprocessing status

```text
mediapipe installed: yes
smoke clips: 20
ok: 20
failed: 0
blocked: 0
decision: READY_FOR_FULL_GENERATION
full generation: pending VM
```

## Requirements

- `PYTHONIOENCODING=utf-8` para compileall en Windows/paths Unicode.
- `faster-whisper` preferido para ASR2; fallback `openai-whisper`.
- `visual_preprocessing/requirements.txt` para MediaPipe.
- GPU VM con datos sincronizados desde `gs://labios-argentos-vsr-data`.

## Commands

Ver `evaluation/experiments/batch_vsr/commands_vm_batch.md`.
