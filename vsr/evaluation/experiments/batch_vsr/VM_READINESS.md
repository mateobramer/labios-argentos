# VM readiness batch VSR

> Gate de verificación en VM con una muestra chica (n=60) para E0–E4 (variantes de
> limpieza de transcripción y preprocesamiento). No se escaló a una corrida completa
> ni se reconcilió con el índice oficial de experimentos
> ([`docs/experiments/`](../../../../docs/experiments/README.md)); se conserva como
> evidencia exploratoria.

## Experimentos

| Experimento | status | train_split | val_split | test_split | rois_root | transcripts_root |
|---|---|---|---|---|---|---|
| E0_baseline_original | READY | `vsr/evaluation/outputs/visual_cleaning/splits_original/train.csv` | `vsr/evaluation/outputs/visual_cleaning/splits_original/val.csv` | `vsr/evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `data/processed/lip_rois` | columna `texto` |
| E1_visual_cleaned | READY | `vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned/train.csv` | `vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned/val.csv` | `vsr/evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `data/processed/lip_rois` | columna `texto` |
| E2_transcript_audited | READY | `vsr/evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/train.csv` | `vsr/evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/val.csv` | `vsr/evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `data/processed/lip_rois` | `vsr/evaluation/outputs/batch_vsr/transcripts_cleaned_stronger` |
| E3_preprocessing_variant | READY | `vsr/evaluation/outputs/visual_cleaning/splits_original/train.csv` | `vsr/evaluation/outputs/visual_cleaning/splits_original/val.csv` | `vsr/evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `vsr/evaluation/outputs/batch_vsr/rois_lower_face_resized96` | columna `texto` |
| E4_all_combined | READY | `vsr/evaluation/outputs/batch_vsr/splits_all_combined/train.csv` | `vsr/evaluation/outputs/batch_vsr/splits_all_combined/val.csv` | `vsr/evaluation/outputs/visual_cleaning/manifests/original_test.csv` | `vsr/evaluation/outputs/batch_vsr/rois_lower_face_resized96` | `vsr/evaluation/outputs/batch_vsr/transcripts_cleaned_stronger` |

## Transcript status

```text
auto_clean changes: 52
ASR2 status: ok 5950/5950
ASR2 model: large-v3-turbo
disagreement rows: 5950
disagreement low: 3784
disagreement medium: 1974
disagreement high: 192
replacement candidates: 2194
auto replacements: 0
usable: 3780
questionable: 1978
bad_candidate: 192
excluded_train: 134
decision: READY_FOR_VM
```

E2 queda habilitado porque ASR2 completo detecta disagreement high y la policy excluye
`bad_candidate` solo de train, manteniendo `questionable`.

## Preprocessing status

```text
mediapipe installed: yes
smoke clips: 20
ok: 20
failed: 0
blocked: 0
decision smoke: READY_FOR_FULL_GENERATION
full generation rows: 5950
full ok: 5950
fallback_original_roi_after_variant_no_frames: 14
decision full: READY
```

Los 14 fallbacks estan marcados en `preprocessing_variant_manifest_full.csv`; copian el
ROI original para evitar huecos cuando la variante no produce frames.

## VM batch ejecutado

VM: `labios-vsr-gpu`, zona `us-east1-d`, GPU NVIDIA L4.

Corrida de verificacion finita:

```text
epochs: 2
batch: 2
accum: 2
test subset: 60 clips deterministico, balanceado por fuente
```

Resultados parseados en `vsr/evaluation/outputs/batch_vsr/results/summary.csv`:

| Experimento | rows | WER | CER |
|---|---:|---:|---:|
| E0_baseline_original | 60 | 0.790218 | 0.482433 |
| E1_visual_cleaned | 60 | 0.796308 | 0.483526 |
| E2_transcript_cleaned_stronger | 60 | 0.757031 | 0.463850 |
| E3_preprocessing_variant | 60 | 0.792764 | 0.492735 |
| E4_all_combined | 60 | 0.791125 | 0.506258 |

Lectura corta: en esta corrida corta E2 mejora contra E0; E1, E3 y E4 no mejoran.
No interpretar estos numeros como convergencia final: son un gate de verificacion VM.

## Requirements

- `PYTHONIOENCODING=utf-8` para compileall en Windows/paths Unicode.
- `faster-whisper` preferido para ASR2; fallback `openai-whisper`.
- `preprocessing/requirements.txt` para MediaPipe.
- GPU VM con datos sincronizados desde `gs://labios-argentos-vsr-data`.

## Commands

Ver `vsr/evaluation/experiments/batch_vsr/commands_vm_batch.md`.
