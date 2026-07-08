# Full clean release report

bucket: gs://labios-argentos-vsr-clean-v1/
branch: feature/full-clean-release

## Argentina existing
- total clips manifest: 9191
- sources: 61
- clean_status_counts: {'baseline_existing_only': 4557, 'blocked_alignment_failed': 729, 'completed_large_turbo_no_gpt': 2193, 'blocked_source_not_found': 1712}
- asr_status_counts: {'pending_reconstruction_or_asr': 4557, 'blocked_alignment_failed': 729, 'completed_large_turbo': 2193, 'blocked_source_not_found': 1712}
- alignment_confidence_counts: {'high': 3286, 'medium': 2479, 'low': 832, 'none': 2594}
- reconstructed_audio_clips: 2193
- large_turbo_asr_rows: 4386
- disagreement_rows: 2193

## Argentina new discovery
- accepted videos queued: 20
- final manifest rows: 12027
- clean_status_counts: {'needs_review': 9779, 'completed_large_turbo_no_gpt': 2248}
- asr_status_counts: {'completed_large_turbo': 11971, 'pending_new_discovery_asr': 56}
- clips_generated_pending_asr_roi: 17
- completed_large_turbo_roi_no_gpt: 0
- source_downloaded_pending_clips_asr_roi: 1
- blocked_download_failed: 2
- reason for remaining blocked: yt-dlp on the VM requires YouTube login/cookies for accepted URLs; local download flow is now available.

## Spanish general
- rows: 47152
- ASR: blocked_missing_provenance_for_asr

## GPT cleaning
- completed_clean_gpt: 0
- completed_large_turbo_no_gpt: 4441
- baseline_existing_only: 4557
- no GPT patch was applied; no cleaning was invented.
