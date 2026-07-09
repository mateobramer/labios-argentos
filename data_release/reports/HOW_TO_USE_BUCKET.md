# HOW TO USE BUCKET

Bucket:

`gs://labios-argentos-vsr-clean-v1/`

Abrir primero:

`gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv`

## Donde esta cada cosa

- Viejos/existing: `argentina/existing/clips_mp4/`
- Nuevos/new_discovery: `argentina/new_discovery/clips_with_audio/`
- Transcripts new_discovery: `argentina/new_discovery/transcripts/`
- ROIs new_discovery: `argentina/new_discovery/rois_npz/`
- Manifests generales: `manifests/`
- Reportes: `reports/`

## Para entrenar VSR

Usar `final_release_manifest.csv` y filtrar:

- `usable_for_training=true`
- `npz_path` no vacio

El texto recomendado esta en `selected_training_text`. Si
`gpt_status=completed_clean_gpt`, viene de limpieza GPT conservadora.

## Para inspeccionar GPT y ROI

- GPT: `manifests/clean_gpt_manifest.csv`
- ROI: `manifests/new_discovery_roi_manifest.csv`
- Reporte ROI: `reports/new_discovery_roi_report.md`

## Pendiente principal

Quedan 10.945 clips sin ROI valido y 500 clips con jobs GPT manuales faltantes. No
relanzar ROI/VM a ciegas: primero diagnosticar decode/AV1 y probar reencode H264 en
muestra.
