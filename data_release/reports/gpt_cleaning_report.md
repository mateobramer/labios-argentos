# GPT cleaning report

completed_clean_gpt: 0
completed_large_turbo_no_gpt: 4441
baseline_existing_only: 4557
needs_review_or_blocked: 16777

No se aplicaron patches GPT en esta corrida. La regla fue no inventar limpieza sin salida JSONL validada.
Los clips con ASR large/turbo usan `large_text` como selected_training_text.

browser_ready: true
gpt_cleaning_status: completed_large_turbo_no_gpt
failure_reason: no_validated_jsonl_patches_available
notes: ChatGPT estaba accesible, pero no habia runner/aplicador batch en el repo para 2248 clips elegibles con ROI valido; se cerro conservadoramente sin inventar patches.
