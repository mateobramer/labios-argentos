# GPT cleaning report

completed_clean_gpt: 13190
rejected_clean_gpt: 1613
completed_large_turbo_no_gpt: 524
baseline_existing_only: 4557
needs_review_or_blocked: 8670
manual_validation_status_counts: {'validated': 83, 'missing_raw_output': 3}
manual_validation_validated_rows: 14559
manual_validation_rejected_rows: 26
manual_validation_missing_rows: 500
text_cleaned_no_roi: 8967

La regla es no inventar limpieza sin salida JSONL validada.
Los clips con ASR large/turbo usan `large_text` como selected_training_text.
