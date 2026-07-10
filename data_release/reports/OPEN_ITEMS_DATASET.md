# OPEN ITEMS DATASET

Regla de trabajo: si alguien intenta arreglar algo, debe actualizar este archivo con que
intento, fecha, resultado, si logro mejora o no, y paths/evidencia. No dejar cambios
incognito. Si se intenta algo y no funciona, marcar `attempted_no_gain` o `blocked`, no
dejar `pending` silencioso.

Estados permitidos: `pending`, `in_progress`, `attempted_no_gain`, `blocked`, `done`,
`abandoned`.

| item_id | area | status | owner | description | evidence/path | next_step | last_updated |
|---|---|---|---|---|---|---|---|
| ROI-001 | ROI recovery new_discovery | pending | delegated / future ROI owner | Recuperar mas ROIs de 10.945 clips sin ROI valido. | `data_release/reports/new_discovery_roi_report.md`; `data_release/new_discovery_roi_manifest.csv`; `final_release_manifest.csv` con 10.945 `npz_path` vacios. | Diagnosticar con `ffprobe`/codec, reencodear muestra H264 y reintentar ROI en sample antes de una VM completa. | 2026-07-09 |
| ROI-002 | Decode / AV1 diagnosis | pending | future ROI owner | Logs mostraron errores AV1/decode; falta distinguir decode failure vs no-face real. | `data_release/reports/roi_recovery_plan.md`; `data_release/reports/roi_handoff_for_next_owner.md`; estados `blocked_roi_no_face`. | Tomar muestra de clips fallidos, correr `ffprobe`, comparar decode local, reencode H264 y medir si MediaPipe recupera caras. | 2026-07-09 |
| GPT-001 | GPT pending jobs | pending | dataset maintainer | Quedaron jobs manuales faltantes: `video_jgp8WZvtkWU__part_002`, `video_jgp8WZvtkWU__part_003`, `video_jgp8WZvtkWU__part_004`; equivalen a 500 clips pendientes. | `cleaning/gpt_clean_v1/reports/manual_gpt_validation_report.csv`; `cleaning/gpt_clean_v1/raw_outputs/`. | Conseguir los 3 raw JSONL faltantes, validar y aplicar solo si pasan schema. | 2026-07-09 |
| ASR-001 | Existing old ASR large/turbo incomplete | pending | dataset maintainer | Solo un subset de existing tuvo `large/turbo`; muchos MP4 existing son ROI silenciosos, no fuente con audio. Para re-ASR hacen falta fuentes originales con audio. | `data_release/asr_large_turbo_manifest.csv`; `data_release/existing_reconstruction_manifest.csv`; `data_release/reports/full_clean_release_report.md`. | Recuperar fuentes originales con audio, reconstruir clips con audio y recien ahi transcribir. | 2026-07-09 |
| SPLIT-001 | Final training split review | pending | training owner | Revisar que el split final train/eval no mezcle fuentes problematicas ni clips con transcripcion/ROI dudosos. | `data_release/final_train_manifest_clean_gpt_v1.csv`; `data_release/final_eval_manifest_clean_gpt_v1.csv`; `data_release/final_release_manifest.csv`. | Auditar por `source_id`, `gpt_status`, `failure_reason`, `needs_review` y `usable_for_training` antes de entrenar. | 2026-07-09 |
| DOC-001 | README maintenance | in_progress | dataset maintainer | Cualquier cambio futuro en bucket/manifests debe actualizar `README_DATASET.md` y `OPEN_ITEMS_DATASET.md`. | `README_DATASET.md`; `OPEN_ITEMS_DATASET.md`; `data_release/reports/`. | Mantener docs sincronizadas con cada rebuild/subida de manifests. | 2026-07-09 |
