# data/

Este repositorio versiona código, documentación, splits y una muestra mínima de
datos. El dataset completo (clips, videos, ROIs) vive en el bucket del proyecto
(`gs://labios-argentos-vsr-clean-v1`, requiere credenciales GCP) y, para los bytes
históricos exactos, en el tag `dataset-clean-v1` / el commit
`a11f0827666b11c975df4d8c5b0d6014894e8ee8`. Política completa de qué se versiona
y qué no: [`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md).

## Qué vive en este directorio

- `corpus/`: transcripciones Whisper por fuente.
- `metadata/`: manifests chicos (`fuentes.csv`, `lip_preprocessing_manifest.csv`,
  `auditoria_clips_manifest.csv`, muestras de `vsr_eval`).
- `samples/El mensaje de Coscu sobre EL AMOR/`: muestra mínima para smoke tests
  (8 clips mp4+txt). Es la fuente más chica del dataset — alcanza para probar
  `preprocessing/` y el pipeline sin inflar el clone. No es representativa para
  evaluación: para eso están los splits congelados en `vsr/splits/` sobre el
  dataset completo.

## Recuperar lo que no está versionado

Bytes exactos vía git (commit pre-limpieza o tag congelado):

```bash
git checkout a11f0827666b11c975df4d8c5b0d6014894e8ee8 -- data/clips
git checkout a11f0827666b11c975df4d8c5b0d6014894e8ee8 -- dataset
git checkout a11f0827666b11c975df4d8c5b0d6014894e8ee8 -- data/videos
# equivalente vía tag:
git show dataset-clean-v1:<path>
```

Forma release, desde el bucket (requiere credenciales del proyecto GCP):

```bash
gsutil -m cp -r "gs://labios-argentos-vsr-clean-v1/argentina/existing/clips_mp4" ./
gsutil -m cp -r "gs://labios-argentos-vsr-clean-v1/argentina/new_discovery/clips_with_audio" ./
gsutil cp "gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv" ./
```

Guía completa del bucket: [`data_pipeline/release/reports/HOW_TO_USE_BUCKET.md`](../data_pipeline/release/reports/HOW_TO_USE_BUCKET.md).
Conteos de referencia: [`data_pipeline/release/reports/bucket_validation_report.md`](../data_pipeline/release/reports/bucket_validation_report.md).
