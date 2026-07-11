# Bucket manual validation report

generated_at: 2026-07-09T01:08:12Z
bucket: gs://labios-argentos-vsr-clean-v1/
reason: `python data_release/scripts/validate_clean_bucket.py` timed out after 1200 seconds in this local run.
sha256_all_match: true
csv_row_counts_all_match: true

| local | gcs | local_rows | remote_rows | rows_match | sha256_match | local_bytes | remote_bytes |
|---|---|---:|---:|---|---|---:|---:|
| `data_release/final_release_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv` | 22384 | 22384 | True | True | 19656656 | 19656656 |
| `data_release/final_train_manifest_clean_gpt_v1.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/final_train_manifest_clean_gpt_v1.csv` | 10315 | 10315 | True | True | 9597171 | 9597171 |
| `data_release/final_eval_manifest_clean_gpt_v1.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/final_eval_manifest_clean_gpt_v1.csv` | 1124 | 1124 | True | True | 900279 | 900279 |
| `data_release/clean_gpt_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/clean_gpt_manifest.csv` | 22384 | 22384 | True | True | 8629510 | 8629510 |
| `data_release/new_discovery_roi_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/new_discovery_roi_manifest.csv` | 13193 | 13193 | True | True | 5544055 | 5544055 |
| `data_release/new_discovery_asr_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/new_discovery_asr_manifest.csv` | 26386 | 26386 | True | True | 16650268 | 16650268 |
| `data_release/new_discovery_clip_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/new_discovery_clip_manifest.csv` | 13706 | 13706 | True | True | 8360366 | 8360366 |
| `data_release/new_discovery_ingest_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/manifests/new_discovery_ingest_manifest.csv` | 20 | 20 | True | True | 11962 | 11962 |
| `data_release/final_release_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/argentina/combined/manifests/final_release_manifest.csv` | 22384 | 22384 | True | True | 19656656 | 19656656 |
| `data_release/clean_gpt_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/argentina/combined/manifests/clean_gpt_manifest.csv` | 22384 | 22384 | True | True | 8629510 | 8629510 |
| `data_release/new_discovery_roi_manifest.csv` | `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/manifests/new_discovery_roi_manifest.csv` | 13193 | 13193 | True | True | 5544055 | 5544055 |
| `data_release/reports/README_DATASET.md` | `gs://labios-argentos-vsr-clean-v1/reports/README_DATASET.md` |  |  |  | True | 4938 | 4938 |
| `data_release/reports/OPEN_ITEMS_DATASET.md` | `gs://labios-argentos-vsr-clean-v1/reports/OPEN_ITEMS_DATASET.md` |  |  |  | True | 3118 | 3118 |
| `data_release/reports/HOW_TO_USE_BUCKET.md` | `gs://labios-argentos-vsr-clean-v1/reports/HOW_TO_USE_BUCKET.md` |  |  |  | True | 1115 | 1115 |
| `data_release/reports/full_clean_release_report.md` | `gs://labios-argentos-vsr-clean-v1/reports/full_clean_release_report.md` |  |  |  | True | 1811 | 1811 |
| `data_release/reports/gpt_cleaning_report.md` | `gs://labios-argentos-vsr-clean-v1/reports/gpt_cleaning_report.md` |  |  |  | True | 535 | 535 |
| `data_cleaning_clean_v1/reports/manual_gpt_validation_report.csv` | `gs://labios-argentos-vsr-clean-v1/reports/manual_gpt_validation_report.csv` | 86 | 86 | True | True | 52498 | 52498 |
| `data_cleaning_clean_v1/reports/manual_gpt_apply_report.md` | `gs://labios-argentos-vsr-clean-v1/reports/manual_gpt_apply_report.md` |  |  |  | True | 486 | 486 |
