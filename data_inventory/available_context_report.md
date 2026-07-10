# Available context report

Este reporte resume que informacion hay disponible para cada bloque de datos, sin tomar decisiones de entrenamiento.

## Dataset propio en `lip_rois/` + `splits/`

Contiene:

- ROIs `.npz`
- clips `.mp4`
- transcripts `.txt`
- splits `train/val/test`
- manifest con `spk`, `titulo`, `clip`, `n_frames`, `texto`, `npz`

Contexto disponible:

- `titulo`: si
- `source_id` aproximado: `spk`
- `split`: si
- `path al ROI`: si, con remapeo de path local a bucket
- `transcript`: si
- `audio/video clip`: si, por `.mp4`
- URL/canal/timestamps originales: no completos en el bucket

Sirve para:

- entrenar/evaluar VSR con manifests y ROIs;
- revisar transcripts a nivel clip de forma parcial, usando `.mp4` + `.txt`;
- no alcanza por si solo para reconstruir contexto completo de fuente/canal/timestamps.

Falta:

- metadata por clip con URL/canal/timestamps;
- licencia/permiso por fuente;
- manifest que una directamente `gs://.../lip_rois/...` con los splits.

## `config/`

Contiene:

- `candidatos_v2_FINAL.csv` con 32 URLs, hablantes y veredicto visual;
- logs/resultados `.log`, `.wer`, `.inf`, `.txt`;
- scripts `.sh` de corridas previas.

Sirve para:

- contexto de fuentes candidatas;
- trazabilidad parcial de resultados previos;
- no contiene datos audiovisuales ni manifests completos por clip.

Falta:

- asociacion completa de URL/canal/timestamp con cada clip de `lip_rois/`;
- licencia por fuente.

## `models/`

Contiene checkpoints `.pth`:

- `es_remapped.pth`
- `ft05_best.pth`
- `ft05b_best.pth`
- `ft06_best.pth`
- `ft07_best.pth`

Sirve para:

- referencia de modelos/resultados previos;
- no sirve para transcript cleaning ni discovery.

Falta:

- documentacion exacta de datos y configuracion usados por cada checkpoint, salvo logs parciales en `config/`.

## `curriculum_visper/`

Contiene:

- `lip_rois/` con `.mp4`, `.npz` y `.txt`;
- 99,946 archivos observados;
- extensiones: `.txt` 46,991; `.npz` 42,599; `.mp4` 10,356.

Contexto disponible:

- titulo/carpeta: si;
- transcript: si;
- audio/video clip: si;
- ROIs: si, aunque no hay `.npz` para todos los `.txt/.mp4`;
- split: no observado;
- URLs/licencia/procedencia: no observadas en el bucket.

Sirve para:

- posible vsr/curriculum/pretraining externo, sujeto a licencia y procedencia;
- revision de clips puntuales si se confirma origen.

No usar mezclado con argentino propio sin:

- confirmar procedencia;
- confirmar licencia;
- definir proposito;
- crear manifests/splits explicitos.

## Data discovery nuevo

Estado despues de auditar round3:

- candidatos buscados: 337;
- candidatos scoreados con URL: 332;
- videos auditados con samples: 67;
- samples registrados: 201;
- `strong_accept`: 10;
- `accept`: 10;
- `maybe_review`: 31;
- `reject`: 281;
- clips aceptados estimados: 11,332;
- minutos utiles estimados: 814.12;
- fuentes accepted distintas: 19;
- target 12K clips: faltan 668 clips;
- target alternativo de 600-900 minutos utiles: alcanzado.

Sirve para:

- shortlist accionable de nuevas fuentes;
- plan de ingesta posterior, sin ingestar todavia.

No contiene aun:

- full videos;
- clips finales;
- transcripts finales;
- ROIs derivados de esos candidatos nuevos.

## Resumen de uso posible

| Bloque | Training/eval | Transcript cleaning | Contexto |
| --- | --- | --- | --- |
| `lip_rois/` + `splits/` | si | parcial | parcial |
| `config/` | no | no | si |
| `models/` | no, son outputs | no | parcial |
| `curriculum_visper/` | posible pretraining, con licencia pendiente | parcial | insuficiente |
| `data_discovery/outputs` | no directamente | no directamente | si, para proxima ingesta |

## Panorama pendiente

- Inventariar en profundidad `gs://labios-argentos-vsr-data/` solo si se decide usarlo como fuente separada.
- Confirmar licencia/procedencia de `curriculum_visper/`.
- Crear un manifest unificado que remapee splits a paths `gs://...`.
- Asociar por clip URL/canal/timestamps cuando exista metadata externa al bucket.
