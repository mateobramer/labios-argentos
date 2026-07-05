# Ingest plan v1

No ejecutar ingest full sin aprobacion explicita. Este plan ordena candidatos para una primera ingesta controlada.

## Orden recomendado
### 1. 🎤 Curso de Oratoria con Daniel Colombo | 100% Práctico
- url: https://www.youtube.com/watch?v=OrwtPwftIi4
- channel: Daniel Colombo
- source_type: educational
- score: total=96.76, visual=96.32, audio=96.67, context=98.0
- estimated accepted clips: 1023.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_educational|formato_largo|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=OrwtPwftIi4"

### 2. ¿Cómo es el ESPAÑOL ARGENTINO FORMAL? - Argento Podcast #44
- url: https://www.youtube.com/watch?v=vDNy6EN7bIY
- channel: Spanish with Nico
- source_type: podcast
- score: total=92.65, visual=96.1, audio=90.0, context=88.0
- estimated accepted clips: 156.0
- estimated usable minutes: 13.0
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_podcast|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=vDNy6EN7bIY"

### 3. GUILLERMO FRANCELLA, UN LUJO COMO PRIMER INVITADO DE OTRO DÍA PERDIDO || ENTREVISTA COMPLETA
- url: https://www.youtube.com/watch?v=R3f0x1IJvhI
- channel: eltrece
- source_type: interview
- score: total=92.19, visual=86.12, audio=98.43, context=98.0
- estimated accepted clips: 256.0
- estimated usable minutes: 21.3
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=R3f0x1IJvhI"

### 4. Mario Pergolini y un imperdible mano a mano con Andy Kusnetzoff | #Perros2022 Perros de la Calle
- url: https://www.youtube.com/watch?v=j4x2GC1Ztro
- channel: Urbana Play 104.3 FM
- source_type: interview
- score: total=92.18, visual=87.28, audio=96.45, context=98.0
- estimated accepted clips: 677.0
- estimated usable minutes: 56.4
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=j4x2GC1Ztro"

### 5. Cómo Amarse a uno Mismo y Tener Buena Salud Mental: Hábitos de Vida - Gabriel Rolón
- url: https://www.youtube.com/watch?v=ITovsJg-q5c
- channel: Tengo un Plan
- source_type: podcast
- score: total=91.1, visual=88.06, audio=91.57, context=98.0
- estimated accepted clips: 1341.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_largo|formato_podcast|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=ITovsJg-q5c"

### 6. El Método Rebord #56 - Andy Chango
- url: https://www.youtube.com/watch?v=h3HtBhArO1Q
- channel: El Método Rebord
- source_type: podcast
- score: total=90.64, visual=84.25, audio=96.38, context=98.0
- estimated accepted clips: 1228.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|black_frame_rate_alto|formato_largo|formato_podcast|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=h3HtBhArO1Q"

### 7. Marcos Galperin | La vida del emprendedor | Aprender de Grandes #072
- url: https://www.youtube.com/watch?v=EuSM3LscaWI
- channel: Aprender de Grandes
- source_type: podcast
- score: total=87.48, visual=80.21, audio=92.58, context=98.0
- estimated accepted clips: 306.0
- estimated usable minutes: 25.5
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|face_detect_rate_bajo|formato_largo|formato_podcast|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=EuSM3LscaWI"

## Datos a guardar en bucket al ingestar
- metadata de fuente y video.
- video/clips/audio/transcripts si se aprueba la ingesta.
- ROIs derivados despues del preprocesamiento visual.

ROIs solos no alcanzan para discovery o limpieza de transcripts: faltan audio, contexto y metadata.