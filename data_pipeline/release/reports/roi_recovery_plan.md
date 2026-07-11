# ROI recovery plan

> Plan documentado durante la fase de escalado del release; no se ejecutó. Se conserva
> como diagnóstico de por qué 10.945 clips quedaron sin ROI válido.

## Estado en el momento del diagnóstico

- Manifest ROI: `data_pipeline/release/new_discovery_roi_manifest.csv`
- Filas ROI new_discovery: 13193
- ROI valido actual: 2248
- ROI valido reflejado en final_release_manifest: 2248 / 13193
- ROI failed/pending actual: 10945
- Status counts: {'blocked_roi_no_face': 10945, 'completed_roi': 2248}

## Failure reasons

- mediapipe_detect_rate_below_threshold: 10945

## Estimacion recuperable

La corrida previa bloqueo clips con deteccion facial parcial por el umbral conservador de MediaPipe.
La recuperacion estimada se calcula sobre fallidos con `detect_rate` ya observado:

- detect_rate >= 0.75: hasta 82 clips recuperables con bajo riesgo.
- detect_rate >= 0.70: hasta 183 clips recuperables razonables si el warp valida.
- detect_rate >= 0.60: hasta 321 clips recuperables agresivos/review; no aplicar sin QA.

Estrategia elegida: reintentar solo fallidos/pending con umbral menos conservador, sin rehacer `completed_roi`, y mantener `needs_review` para casos dudosos.

## Breakdown por video_id

| video_id | total | completed_roi | failed_pending | reason principal |
| --- | ---: | ---: | ---: | --- |
| h3HtBhArO1Q | 1580 | 0 | 1580 | mediapipe_detect_rate_below_threshold |
| ITovsJg-q5c | 1308 | 79 | 1229 | mediapipe_detect_rate_below_threshold |
| YYIVFA000BI | 1166 | 0 | 1166 | mediapipe_detect_rate_below_threshold |
| qEKPgqURvo0 | 1058 | 0 | 1058 | mediapipe_detect_rate_below_threshold |
| jgp8WZvtkWU | 918 | 0 | 918 | mediapipe_detect_rate_below_threshold |
| OrwtPwftIi4 | 884 | 0 | 884 | mediapipe_detect_rate_below_threshold |
| j4x2GC1Ztro | 832 | 0 | 832 | mediapipe_detect_rate_below_threshold |
| Z-t-GNlxpYc | 805 | 0 | 805 | mediapipe_detect_rate_below_threshold |
| EuSM3LscaWI | 571 | 0 | 571 | mediapipe_detect_rate_below_threshold |
| q94HfK07DjI | 470 | 0 | 470 | mediapipe_detect_rate_below_threshold |
| PRQEkiIWps0 | 407 | 0 | 407 | mediapipe_detect_rate_below_threshold |
| WMk6afYRfKM | 911 | 533 | 378 | mediapipe_detect_rate_below_threshold |
| R3f0x1IJvhI | 293 | 0 | 293 | mediapipe_detect_rate_below_threshold |
| JixCyhEGE0A | 552 | 416 | 136 | mediapipe_detect_rate_below_threshold |
| sjnH4bTak9s | 238 | 128 | 110 | mediapipe_detect_rate_below_threshold |
| EhYznjqlcKY | 393 | 337 | 56 | mediapipe_detect_rate_below_threshold |
| eqw0QM4A0oA | 569 | 519 | 50 | mediapipe_detect_rate_below_threshold |
| yJxDKBgw5NU | 238 | 236 | 2 | mediapipe_detect_rate_below_threshold |

## Top videos con mas fallos

- h3HtBhArO1Q: 1580/1580 failed_pending; principal=mediapipe_detect_rate_below_threshold
- ITovsJg-q5c: 1229/1308 failed_pending; principal=mediapipe_detect_rate_below_threshold
- YYIVFA000BI: 1166/1166 failed_pending; principal=mediapipe_detect_rate_below_threshold
- qEKPgqURvo0: 1058/1058 failed_pending; principal=mediapipe_detect_rate_below_threshold
- jgp8WZvtkWU: 918/918 failed_pending; principal=mediapipe_detect_rate_below_threshold
- OrwtPwftIi4: 884/884 failed_pending; principal=mediapipe_detect_rate_below_threshold
- j4x2GC1Ztro: 832/832 failed_pending; principal=mediapipe_detect_rate_below_threshold
- Z-t-GNlxpYc: 805/805 failed_pending; principal=mediapipe_detect_rate_below_threshold
- EuSM3LscaWI: 571/571 failed_pending; principal=mediapipe_detect_rate_below_threshold
- q94HfK07DjI: 470/470 failed_pending; principal=mediapipe_detect_rate_below_threshold

## Reglas de retry

- No rehacer `completed_roi`.
- Descargar clips desde GCS solo cuando no existe cache local.
- Checkpoint y upload cada 25 clips.
- Registrar estados: `completed_roi`, `blocked_no_face`, `blocked_low_face_ratio`, `blocked_bad_video`, `blocked_short_clip`, `blocked_mouth_not_found`, `needs_review`.
- Publicar heartbeat/status en `gs://labios-argentos-vsr-clean-v1/reports/roi_vm_run_status.json` y `gs://labios-argentos-vsr-clean-v1/reports/roi_vm_heartbeat.txt`.
