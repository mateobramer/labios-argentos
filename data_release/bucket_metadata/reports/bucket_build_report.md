# Bucket build report

bucket_destino: gs://labios-argentos-vsr-clean-v1/
estado: populated_validated_clean_v1_baseline
source_bucket: gs://labios-argentos-vsr-dataset/

No se modifico ni borro el bucket fuente.

## Datos copiados

- argentina/existing/clips_mp4: 12112 `.mp4`
- argentina/existing/rois_npz: 12112 `.npz`
- argentina/existing/transcripts/large: 12112 `.txt`
- argentina/existing/transcripts/clean_v1: 12112 `.txt` baseline conservador
- spanish_general/existing/clips_mp4: 10356 `.mp4`
- spanish_general/existing/rois_npz: 42599 `.npz`
- spanish_general/existing/transcripts/large: 46991 `.txt`

## Manifests y reportes

- argentina/existing/manifests/argentina_existing_manifest.csv
- argentina/new_discovery/manifests/argentina_new_manifest.csv
- spanish_general/existing/manifests/spanish_general_manifest.csv
- argentina/existing/manifests/clean_manifest.csv
- argentina/existing/manifests/asr_manifest.csv
- reports/manifest_build_report.md
- reports/cleaning_report.md
- reports/bucket_validation_report.md
- reports/failures.csv

## Estado ASR/cleaning

Las muestras `.mp4` existentes son ROIs 96x96 sin pista de audio. `turbo` para
datos existentes queda `blocked_no_audio_in_roi_mp4`.

`clean_v1` existe como baseline igual a `large_existing`, marcado en
`clean_manifest.csv` como `unchanged_no_llm_baseline`. No se aplicaron patches GPT.

## Validacion

`data_release/reports/bucket_validation_report.md` confirma:

- IAM `roles/storage.objectViewer` para `fgutman@udesa.edu.ar`.
- 5 `.mp4` descargados/abiertos: video presente, audio ausente.
- 5 `.npz` descargados/abiertos: key `rois`, dtype `uint8`, shape `T x 96 x 96`.
- 5 `.txt` `clean_v1` leidos.
- Sin VMs/discos/IPs estaticas remanentes con nombre `vsr-cleaning-vm`.
