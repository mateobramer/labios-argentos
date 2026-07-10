# Bucket inventory

Bucket principal revisado: `gs://labios-argentos-vsr-dataset/`.

Comandos usados como evidencia:

```bash
gcloud.cmd storage ls gs://labios-argentos-vsr-dataset/
gcloud.cmd storage ls -r gs://labios-argentos-vsr-dataset/config/
gcloud.cmd storage ls -r gs://labios-argentos-vsr-dataset/splits/
gcloud.cmd storage du --summarize gs://labios-argentos-vsr-dataset/config/
gcloud.cmd storage du --summarize gs://labios-argentos-vsr-dataset/splits/
gcloud.cmd storage du --summarize gs://labios-argentos-vsr-dataset/models/
gcloud.cmd storage du --summarize gs://labios-argentos-vsr-dataset/lip_rois/
gsutil.cmd du -s gs://labios-argentos-vsr-dataset/curriculum_visper/
gsutil.cmd ls -r gs://labios-argentos-vsr-dataset/<folder>/**
gsutil.cmd cat gs://labios-argentos-vsr-dataset/splits/splits.csv
gsutil.cmd cat gs://labios-argentos-vsr-dataset/config/candidatos_v2_FINAL.csv
```

`gcloud storage ls -r` sobre carpetas grandes encontro un problema de encoding en Windows con algunos nombres de objeto; para conteos completos se uso `gsutil ls -r` capturado por Python como bytes, sin descargar objetos.

## Estructura principal

| Carpeta | Archivos | Tamano | Contenido |
| --- | ---: | ---: | --- |
| `config/` | 22 | 355,765 bytes | logs, scripts, resultados, candidatos con URLs |
| `splits/` | 4 | 4,561,702 bytes | manifests `train/val/test/splits.csv` |
| `lip_rois/` | 36,336 | 10,948,274,151 bytes | 12,112 tripletas `.mp4/.npz/.txt` |
| `models/` | 5 | 1,051,359,937 bytes | checkpoints `.pth` |
| `curriculum_visper/` | 99,946 | 29,472,204,780 bytes | clips/ROIs/textos bajo `lip_rois/` |

## Splits

`splits/splits.csv` tiene 9,191 filas con columnas:

`split, spk, titulo, clip, n_frames, texto, npz`

Distribucion:

- `train`: 8,067 filas
- `val`: 466 filas
- `test`: 658 filas
- titulos/speakers: 61

Los paths de `npz` apuntan a `data/processed/lip_rois/...`, por lo que para usar el bucket hay que remapearlos a `gs://labios-argentos-vsr-dataset/lip_rois/...`.

## Datos audiovisuales y contexto

`lip_rois/` contiene clip `.mp4`, ROI `.npz` y transcript `.txt` por clip. Eso alcanza para entrenar/evaluar VSR si los paths se remapean correctamente con `splits/`.

Para transcript cleaning, el bucket principal alcanza solo parcialmente:

- tiene clip `.mp4` y `.txt`;
- no tiene URL/canal/timestamps por clip dentro de `lip_rois/`;
- `config/candidatos_v2_FINAL.csv` aporta 32 URLs candidatas, pero no una metadata completa por clip.

`curriculum_visper/` tiene datos parecidos en volumen mucho mayor, pero no trae split, licencia, URLs ni procedencia suficiente en el bucket. Tratarlo como vsr/curriculum/pretraining externo o sensible hasta documentar origen y permisos.

## Bucket separado observado

Tambien existe `gs://labios-argentos-vsr-data/` con top-level:

- `adaptacion/`
- `corrector/`
- `lip_rois/`
- `lip_rois_full/`
- `models/`

No se mezcla con el bucket principal en este inventario.
