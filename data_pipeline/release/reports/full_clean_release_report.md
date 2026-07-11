# Full clean release report

bucket: gs://labios-argentos-vsr-clean-v1/
branch: feature/full-clean-release

## Argentina existing
- total clips manifest: 9191
- sources: 61
- clean_status_counts: {'baseline_existing_only': 4557, 'blocked_alignment_failed': 729, 'completed_clean_gpt': 2142, 'completed_large_turbo_no_gpt': 51, 'blocked_source_not_found': 1712}
- asr_status_counts: {'pending_reconstruction_or_asr': 4557, 'blocked_alignment_failed': 729, 'completed_large_turbo': 2193, 'blocked_source_not_found': 1712}
- alignment_confidence_counts: {'high': 3286, 'medium': 2479, 'low': 832, 'none': 2594}
- reconstructed_audio_clips: 2193
- large_turbo_asr_rows: 4386
- disagreement_rows: 2193

## Argentina new discovery
- accepted videos queued: 20
- final manifest rows: 13193
- clean_status_counts: {'completed_clean_gpt': 11048, 'completed_large_turbo_no_gpt': 2086, 'needs_review': 59}
- asr_status_counts: {'completed_large_turbo': 13134, 'pending_new_discovery_asr': 59}
- clips_generated_pending_asr_roi: 18
- completed_large_turbo_roi_no_gpt: 0
- source_downloaded_pending_clips_asr_roi: 2
- blocked_download_failed: 0
- reason for remaining blocked: yt-dlp on the VM requires YouTube login/cookies for accepted URLs; local download flow is now available.

## Spanish general
- rows: 47152
- ASR: blocked_missing_provenance_for_asr

## GPT cleaning
- completed_clean_gpt: 13190
- rejected_clean_gpt: 1613
- completed_large_turbo_no_gpt: 524
- baseline_existing_only: 4557
- manual_validation_status_counts: {'validated': 83, 'missing_raw_output': 3}
- manual_validation_validated_rows: 14559
- manual_validation_rejected_rows: 26
- manual_validation_missing_rows: 500
- text_cleaned_no_roi: 8967
- GPT patches are applied only from validated JSONL outputs.
