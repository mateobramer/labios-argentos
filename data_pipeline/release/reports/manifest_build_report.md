# Manifest build report

source_bucket: `gs://labios-argentos-vsr-dataset`
bucket_inventory_csv: Disponible

## Argentina existing
- filas en splits: 9191
- tripletas encontradas: 9191
- filas con algun archivo faltante: 0
- archivos en bucket no usados por splits: 8763
- distribucion split: {'test': 658, 'train': 8067, 'val': 466}
- URL mapeada/sin URL: {'missing': 5047, 'mapped': 4144}

## Spanish general
- filas manifest: 47152
- archivos listados: 99946
- extensiones: {'.mp4': 10356, '.npz': 42599, '.txt': 46991}
- tripletas completas: 5968
- licencia/procedencia: unknown_or_sensitive / not_documented_in_bucket

## Argentina new discovery
- videos accepted: 20
- decisiones: {'strong_accept': 10, 'accept': 10}

## Target progress snapshot
```
# Target progress v1

target_new_accepted_clips: 12000
stretch_new_accepted_clips: 20000
accepted_new_clips_estimate: 11332
remaining_clips_to_target: 668
usable_minutes_estimate: 814.1
remaining_usable_minutes: 0.0
target_clips_reached: false
target_usable_minutes_reached: true
accepted_videos_count: 20
maybe_videos_count: 31
rejected_videos_count: 281
distinct_sources_count: 19
clips_per_minute_estimate: 12.00
source_caps_applied: max_clips_per_source=1800, max_usable_minutes_per_source=60
quality_thresholds: min_total_score=85, min_visual_quality_score=80, min_audio_quality_score=75, accent=argentino/rioplatense_probable
target_decision: TARGET_CLIPS_NOT_REACHED_USABLE_MINUTES_REACHED

## Top sources
- Tengo un Plan: 1341.0 clips, 60.0 min, decision=cap_applied_or_near_cap
- El Método Rebord: 1228.0 clips, 60.0 min, decision=cap_applied_or_near_cap
- Daniel Colombo: 1023.0 clips, 60.0 min, decision=cap_applied_or_near_cap
- BLENDER: 822.0 clips, 60.0 min, decision=cap_applied_or_near_cap
- Cenital: 748.0 clips, 60.0 min, decision=cap_applied_or_near_cap
- Todo Noticias: 701.0 clips, 58.41 min, decision=ok
- Perfil: 682.0 clips, 56.86 min, decision=ok
- Alberto Fernández: 680.0 clips, 56.7 min, decision=ok
- Urbana Play 104.3 FM: 677.0 clips, 56.41 min, decision=ok
- Futurock FM: 598.0 clips, 49.83 min, decision=ok

## Nota de scope
La shortlist accepted no llegó a ingestarse completa; los backups `maybe_review` no se usaron.
```
