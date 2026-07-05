# Ingest plan v1

No ejecutar ingest full sin aprobacion explicita. Este plan ordena candidatos para una primera ingesta controlada.

## Orden recomendado
### 1. Santiago Cafiero, mano a mano en TN: "Es una decisión sanitaria, no política"
- url: https://www.youtube.com/watch?v=yJxDKBgw5NU
- channel: Todo Noticias
- source_type: interview
- score: total=97.28, visual=100.0, audio=92.28, context=98.0
- estimated accepted clips: 248.0
- estimated usable minutes: 20.6
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=yJxDKBgw5NU"

### 2. 🎤 Curso de Oratoria con Daniel Colombo | 100% Práctico
- url: https://www.youtube.com/watch?v=OrwtPwftIi4
- channel: Daniel Colombo
- source_type: educational
- score: total=96.76, visual=96.32, audio=96.67, context=98.0
- estimated accepted clips: 1023.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_educational|formato_largo|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=OrwtPwftIi4"

### 3. Javier Milei en TN I Entrevista completa del 30/06/2024
- url: https://www.youtube.com/watch?v=JixCyhEGE0A
- channel: Todo Noticias
- source_type: interview
- score: total=93.47, visual=90.3, audio=95.72, context=98.0
- estimated accepted clips: 453.0
- estimated usable minutes: 37.8
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=JixCyhEGE0A"

### 4. ¿Cómo es el ESPAÑOL ARGENTINO FORMAL? - Argento Podcast #44
- url: https://www.youtube.com/watch?v=vDNy6EN7bIY
- channel: Spanish with Nico
- source_type: podcast
- score: total=92.65, visual=96.1, audio=90.0, context=88.0
- estimated accepted clips: 156.0
- estimated usable minutes: 13.0
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_podcast|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=vDNy6EN7bIY"

### 5. GUILLERMO FRANCELLA, UN LUJO COMO PRIMER INVITADO DE OTRO DÍA PERDIDO || ENTREVISTA COMPLETA
- url: https://www.youtube.com/watch?v=R3f0x1IJvhI
- channel: eltrece
- source_type: interview
- score: total=92.19, visual=86.12, audio=98.43, context=98.0
- estimated accepted clips: 256.0
- estimated usable minutes: 21.3
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=R3f0x1IJvhI"

### 6. Mario Pergolini y un imperdible mano a mano con Andy Kusnetzoff | #Perros2022 Perros de la Calle
- url: https://www.youtube.com/watch?v=j4x2GC1Ztro
- channel: Urbana Play 104.3 FM
- source_type: interview
- score: total=92.18, visual=87.28, audio=96.45, context=98.0
- estimated accepted clips: 677.0
- estimated usable minutes: 56.4
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=j4x2GC1Ztro"

### 7. Entrevista completa de Javier Milei con Luis Majul: "Hemos pasado el momento bisagra"
- url: https://www.youtube.com/watch?v=Z-t-GNlxpYc
- channel: LA NACION
- source_type: interview
- score: total=91.18, visual=90.66, audio=87.51, context=98.0
- estimated accepted clips: 596.0
- estimated usable minutes: 49.6
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_proxy_intermedio|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=Z-t-GNlxpYc"

### 8. Cómo Amarse a uno Mismo y Tener Buena Salud Mental: Hábitos de Vida - Gabriel Rolón
- url: https://www.youtube.com/watch?v=ITovsJg-q5c
- channel: Tengo un Plan
- source_type: podcast
- score: total=91.1, visual=88.06, audio=91.57, context=98.0
- estimated accepted clips: 1341.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_largo|formato_podcast|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=ITovsJg-q5c"

### 9. Taty Almeida | Bios Militantes con Julia Mengolini en #Segurola
- url: https://www.youtube.com/watch?v=eqw0QM4A0oA
- channel: Futurock FM
- source_type: podcast
- score: total=90.38, visual=87.56, audio=90.0, context=98.0
- estimated accepted clips: 598.0
- estimated usable minutes: 49.8
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_largo|formato_podcast|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=eqw0QM4A0oA"

### 10. Martin Menem con Iván Schargrodsky en #OnTheRecord
- url: https://www.youtube.com/watch?v=jgp8WZvtkWU
- channel: Cenital
- source_type: interview
- score: total=90.69, visual=95.53, audio=77.77, context=98.0
- estimated accepted clips: 748.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: medium:samples_incompletos|audio_asr_confirmado; audio_proxy_intermedio|audio_proxy_suficiente|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=jgp8WZvtkWU"

### 11. El Método Rebord #56 - Andy Chango
- url: https://www.youtube.com/watch?v=h3HtBhArO1Q
- channel: El Método Rebord
- source_type: podcast
- score: total=90.64, visual=84.25, audio=96.38, context=98.0
- estimated accepted clips: 1228.0
- estimated usable minutes: 60.0
- recommended_use: ingest_partial
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|black_frame_rate_alto|formato_largo|formato_podcast|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados|source_cap_applied_partial
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=h3HtBhArO1Q"

### 12. Obstáculos ideológicos en el empleo
- url: https://www.youtube.com/watch?v=sjnH4bTak9s
- channel: Universidad Torcuato Di Tella
- source_type: interview
- score: total=88.85, visual=82.07, audio=94.06, context=98.0
- estimated accepted clips: 110.0
- estimated usable minutes: 9.1
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=sjnH4bTak9s"

### 13. Marcos Galperin | La vida del emprendedor | Aprender de Grandes #072
- url: https://www.youtube.com/watch?v=EuSM3LscaWI
- channel: Aprender de Grandes
- source_type: podcast
- score: total=87.48, visual=80.21, audio=92.58, context=98.0
- estimated accepted clips: 306.0
- estimated usable minutes: 25.5
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|face_detect_rate_bajo|formato_largo|formato_podcast|fuente_argentina_probable|speaker_count_usable|total_score_alto|visual_proxy_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=EuSM3LscaWI"

### 14. ENTREVISTA: DIEGO TORRES
- url: https://www.youtube.com/watch?v=IGYG0Kn0wxo
- channel: Congo
- source_type: interview
- score: total=86.56, visual=91.29, audio=94.39, context=63.0
- estimated accepted clips: 374.0
- estimated usable minutes: 31.2
- recommended_use: ingest_full
- riesgos: low; audio_proxy_alto|audio_score_sobre_threshold|contexto_incierto|formato_a_evitar_keyword|formato_interview|formato_largo|fuente_argentina_probable|scene_cut_proxy_alto|speaker_count_usable|total_score_alto|visual_score_sobre_threshold|asr_espanol_y_habla_confirmados
- comando tentativo: python descargar_procesar.py "https://www.youtube.com/watch?v=IGYG0Kn0wxo"

## Datos a guardar en bucket al ingestar
- metadata de fuente y video.
- video/clips/audio/transcripts si se aprueba la ingesta.
- ROIs derivados despues del preprocesamiento visual.

ROIs solos no alcanzan para discovery o limpieza de transcripts: faltan audio, contexto y metadata.

## Comandos para escalar discovery
No buscar llegar a 12K en un loop local largo. Repetir en lotes chicos, guardando outputs livianos y frenando cualquier auditoria sin progreso/logs nuevos por mas de 30 minutos.

```bash
python -m data_discovery.src.search_youtube_candidates --append --max-results 8 --query "podcast argentino entrevista camara fija" --query "entrevista argentina completa"
python -m data_discovery.src.audit_youtube_candidate --input data_discovery/outputs/candidate_videos_round3_for_audit.csv --limit 15 --run-asr --asr-model small
python -m data_discovery.src.score_candidates --clips-per-minute 12
python -m data_discovery.src.make_contact_sheets --accepted-only --limit 80
```