# ROI recovery — diagnóstico de cierre

> La recuperación de ROI se detuvo en este punto y no se retomó. Se conserva como
> diagnóstico de la causa probable (decode/AV1), no como una tarea pendiente.

## Resumen

- Bucket: `gs://labios-argentos-vsr-clean-v1/`
- ROI valido final actual: 2248 clips.
- ROI failed/pending actual: 10945 clips.
- Estado autoritativo local: `data_pipeline/release/new_discovery_roi_manifest.csv`.
- La corrida de recovery no recupero ROIs adicionales; se detiene la recuperacion por ahora.
- No correr retry masivo sin diagnosticar decode/reencode primero.

## Manifests y reportes relevantes en bucket

- `gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv`
- `gs://labios-argentos-vsr-clean-v1/manifests/new_discovery_roi_manifest.csv`
- `gs://labios-argentos-vsr-clean-v1/manifests/new_discovery_clip_manifest.csv`
- `gs://labios-argentos-vsr-clean-v1/reports/new_discovery_roi_report.md`
- `gs://labios-argentos-vsr-clean-v1/reports/roi_recovery_plan.md`
- `gs://labios-argentos-vsr-clean-v1/reports/roi_handoff_for_next_owner.md`

## Conteos actuales

Conteos desde `data_pipeline/release/new_discovery_roi_manifest.csv`:

| status | clips |
| --- | ---: |
| completed_roi | 2248 |
| blocked_roi_no_face | 10945 |

El script de recovery nuevo tambien contempla estados mas finos (`blocked_no_face`, `blocked_low_face_ratio`, `blocked_bad_video`, `blocked_short_clip`, `blocked_mouth_not_found`, `needs_review`), pero el manifest final disponible para handoff todavia conserva los fallos como `blocked_roi_no_face`.

Failure note dominante:

| notes | clips |
| --- | ---: |
| mediapipe_detect_rate_below_threshold | 10945 |

## Observaciones de recovery

- La recovery VM se uso solo para ROI, sin ASR.
- La corrida no aumento el total de `completed_roi`; el ROI valido final queda en 2248.
- Se observaron problemas de decode/AV1 durante la corrida y muchos clips terminaron sin deteccion util de cara/boca (`blocked_roi_no_face` / no face). Esta observacion viene de logs operacionales de la corrida; los conteos autoritativos estan en el manifest ROI.
- Antes de cualquier retry completo, hacer una muestra diagnostica con decode/reencode: verificar que OpenCV/ffmpeg lea frames reales, reencodear a H.264 si hace falta, y recien despues probar MediaPipe/landmarks.

## Top videos con mas fallos

| video_id | total clips | completed_roi | failed/pending |
| --- | ---: | ---: | ---: |
| h3HtBhArO1Q | 1580 | 0 | 1580 |
| ITovsJg-q5c | 1308 | 79 | 1229 |
| YYIVFA000BI | 1166 | 0 | 1166 |
| qEKPgqURvo0 | 1058 | 0 | 1058 |
| jgp8WZvtkWU | 918 | 0 | 918 |
| OrwtPwftIi4 | 884 | 0 | 884 |
| j4x2GC1Ztro | 832 | 0 | 832 |
| Z-t-GNlxpYc | 805 | 0 | 805 |
| EuSM3LscaWI | 571 | 0 | 571 |
| q94HfK07DjI | 470 | 0 | 470 |
| PRQEkiIWps0 | 407 | 0 | 407 |
| WMk6afYRfKM | 911 | 533 | 378 |
| R3f0x1IJvhI | 293 | 0 | 293 |
| JixCyhEGE0A | 552 | 416 | 136 |
| sjnH4bTak9s | 238 | 128 | 110 |

## Donde estan los clips y fuentes

Usar `new_discovery_clip_manifest.csv` para ubicar insumos:

- Clips con audio por clip: `clip_video_gcs_path`, bajo `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio/<video_id>/clip_NNNN.mp4`.
- Video fuente descargado: `source_video_gcs_path`, bajo `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_videos/<video_id>/`.
- Audio fuente descargado: `source_audio_gcs_path`, bajo `gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/source_audio/<video_id>/`.

Los originales y clips de entrada no deben tocarse ni sobrescribirse. Cualquier retry debe escribir nuevos outputs ROI y un nuevo manifest/versionado.

## Columnas utiles para retry

Desde `new_discovery_roi_manifest.csv`:

- `video_id`
- `clip_id`
- `clip_name`
- `clip_video_gcs_path`
- `detect_rate`
- `frames`
- `status`
- `notes`

Desde `new_discovery_clip_manifest.csv`:

- `source_video_gcs_path`
- `source_audio_gcs_path`
- `start_time`
- `end_time`
- `duration`
- `clip_video_gcs_path`
- `status`

## Sugerencia de proximo intento

1. Tomar una muestra de videos top failed, empezando por `h3HtBhArO1Q`, `YYIVFA000BI` y `qEKPgqURvo0`.
2. Descargar pocos clips y confirmar decode frame-by-frame con ffmpeg y OpenCV.
3. Si hay warnings AV1/decode o frames vacios, reencodear la muestra a H.264/AAC antes de MediaPipe.
4. Medir detect_rate despues del reencode.
5. Solo si la muestra mejora, correr retry masivo con manifest nuevo.
6. Mantener intactos `source_videos`, `source_audio` y `clips_with_audio` originales.
