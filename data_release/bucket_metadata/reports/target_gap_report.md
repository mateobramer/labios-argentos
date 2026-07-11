# Target gap report

bucket: gs://labios-argentos-vsr-clean-v1/
branch: feature/full-clean-release
target_definition: existing official + new_discovery accepted estimates, not 12k total

## Summary

- existing_official_count: 9191
- old_physical_bucket_count: 12112
- new_discovery_estimated_total: 11332
- new_discovery_currently_in_manifest: 2888
- new_discovery_with_large_turbo: 2885
- new_discovery_with_valid_roi: 79
- remaining_new_discovery_estimated: 8796
- target_min_total_final_release: 20523
- current_final_release_manifest_rows: 12079
- current_gap_to_target_min_total: 8444

## Accepted videos

| video_id | title | decision | accepted_clips_estimate | current_clips | current_large_turbo | current_valid_roi | remaining_estimate | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| yJxDKBgw5NU | Santiago Cafiero, mano a mano en TN: "Es una decisión sanitaria, no política" | strong_accept | 248 | 0 | 0 | 0 | 248 | blocked_youtube_requires_login_or_cookies_from_vm |
| OrwtPwftIi4 | 🎤 Curso de Oratoria con Daniel Colombo / 100% Práctico | strong_accept | 1023 | 0 | 0 | 0 | 1023 | blocked_youtube_requires_login_or_cookies_from_vm |
| qEKPgqURvo0 | REBORD - PERETTI / HAY ALGO AHÍ / BLENDER | strong_accept | 822 | 0 | 0 | 0 | 822 | blocked_youtube_requires_login_or_cookies_from_vm |
| JixCyhEGE0A | Javier Milei en TN I Entrevista completa del 30/06/2024 | strong_accept | 453 | 0 | 0 | 0 | 453 | blocked_youtube_requires_login_or_cookies_from_vm |
| vDNy6EN7bIY | ¿Cómo es el ESPAÑOL ARGENTINO FORMAL? - Argento Podcast #44 | strong_accept | 156 | 0 | 0 | 0 | 156 | blocked_youtube_requires_login_or_cookies_from_vm |
| R3f0x1IJvhI | GUILLERMO FRANCELLA, UN LUJO COMO PRIMER INVITADO DE OTRO DÍA PERDIDO // ENTREVISTA COMPLETA | strong_accept | 256 | 0 | 0 | 0 | 256 | blocked_youtube_requires_login_or_cookies_from_vm |
| j4x2GC1Ztro | Mario Pergolini y un imperdible mano a mano con Andy Kusnetzoff / #Perros2022 Perros de la Calle | strong_accept | 677 | 0 | 0 | 0 | 677 | blocked_youtube_requires_login_or_cookies_from_vm |
| Z-t-GNlxpYc | Entrevista completa de Javier Milei con Luis Majul: "Hemos pasado el momento bisagra" | strong_accept | 596 | 0 | 0 | 0 | 596 | blocked_youtube_requires_login_or_cookies_from_vm |
| ITovsJg-q5c | Cómo Amarse a uno Mismo y Tener Buena Salud Mental: Hábitos de Vida - Gabriel Rolón | strong_accept | 1341 | 1308 | 1308 | 79 | 33 | partial_clips_generated_pending_asr_roi |
| eqw0QM4A0oA | Taty Almeida / Bios Militantes con Julia Mengolini en #Segurola | strong_accept | 598 | 0 | 0 | 0 | 598 | blocked_youtube_requires_login_or_cookies_from_vm |
| jgp8WZvtkWU | Martin Menem con Iván Schargrodsky en #OnTheRecord | accept | 748 | 0 | 0 | 0 | 748 | blocked_youtube_requires_login_or_cookies_from_vm |
| h3HtBhArO1Q | El Método Rebord #56 - Andy Chango | accept | 1228 | 1580 | 1577 | 0 | 0 | processed_full_estimate_roi_evaluated |
| EhYznjqlcKY | Guido Di Tella con Juan Carlos de Pablo - DiFilm (1998) | accept | 408 | 0 | 0 | 0 | 408 | blocked_youtube_requires_login_or_cookies_from_vm |
| YYIVFA000BI | Pepe Mujica con Jorge Fontevecchia (Entrevista Completa) | accept | 682 | 0 | 0 | 0 | 682 | blocked_youtube_requires_login_or_cookies_from_vm |
| WMk6afYRfKM | Entrevista en “IP Noticias” con Noelia Barral, Romina Calderaro y Nora Veiras - IP - 07/02/2021 | accept | 680 | 0 | 0 | 0 | 680 | blocked_youtube_requires_login_or_cookies_from_vm |
| PRQEkiIWps0 | Coscu Mete Un Bombazo Histórico Después De Probar Las Salsas Más Picantes Del Mundo | accept | 331 | 0 | 0 | 0 | 331 | blocked_youtube_requires_login_or_cookies_from_vm |
| sjnH4bTak9s | Obstáculos ideológicos en el empleo | accept | 110 | 0 | 0 | 0 | 110 | blocked_youtube_requires_login_or_cookies_from_vm |
| q94HfK07DjI | El último reportaje del papa Francisco con Infobae | accept | 295 | 0 | 0 | 0 | 295 | blocked_youtube_requires_login_or_cookies_from_vm |
| EuSM3LscaWI | Marcos Galperin / La vida del emprendedor / Aprender de Grandes #072 | accept | 306 | 0 | 0 | 0 | 306 | blocked_youtube_requires_login_or_cookies_from_vm |
| IGYG0Kn0wxo | ENTREVISTA: DIEGO TORRES | accept | 374 | 0 | 0 | 0 | 374 | blocked_youtube_requires_login_or_cookies_from_vm |

## Processing order

1. Process accepted/strong_accept rows with source_video_gcs_path/source_audio_gcs_path already present and incomplete.
2. Download accepted/strong_accept rows without raw source locally with browser cookies, then upload source_video/source_audio/metadata to GCS.
3. Run VM GPU normal from GCS with resume/checkpoints; do not stop at 12k total.
4. Run GPT cleaning only after new_discovery processing completes, and only for clips with large+turbo+valid ROI.
