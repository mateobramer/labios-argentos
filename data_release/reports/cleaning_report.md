# Cleaning report clean_v1

argentina_existing_manifest_rows: 9191
clean_manifest_rows: 9191
context_pack_sources: 61
status_counts: {'unchanged_no_llm_baseline': 9191}
argentina_new_accepted_videos_queued: 20

## ASR status

- Existing argentina ROI mp4 probe: video-only 96x96, audio stream absent.
- Existing spanish_general ROI mp4 probe: video-only 96x96, audio stream absent.
- Turbo ASR for existing data is blocked until clips with audio are reconstructed.

## clean_v1 status

`clean_v1` mirrors `large_existing` as a conservative baseline.
No GPT patch was applied in this run, and every row is marked
`unchanged_no_llm_baseline` with low confidence plus review reason.

## Next scaling step

Reconstruct audio-bearing clips from mapped URLs/raw sources, run large/turbo,
then feed the generated context packs to the JSONL prompt and validate patches
before replacing clean_v1 text.
