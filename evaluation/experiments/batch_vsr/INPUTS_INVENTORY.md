# Inventario minimo de inputs batch VSR

Fecha de inspeccion local: 2026-07-04.

## Clips `.mp4` originales

- Ruta: `data/clips/<titulo>/<clip>.mp4`.
- Conteo local observado: 8547 `.mp4`.
- Uso: fuente para regenerar ROIs y, si hiciera falta, extraer audio.

## ROIs `.npz` actuales

- Ruta principal: `data/processed/lip_rois/<titulo>/<clip>.npz`.
- Conteo local observado: 5968 `.npz`.
- Los splits VSR usan 5950 filas en `vsr_models/splits/splits.csv`.
- Formato esperado por `fine_tune.py`: array `rois` cargado desde `.npz`.

## Transcripts `.txt`

- Ruta de clips crudos: `data/clips/<titulo>/<clip>.txt`.
- Conteo local observado: 8547 `.txt`.
- `vsr_models/splits/splits.csv` tambien incluye la columna `texto`; ese es el label que
  usaba `fine_tune.py` antes de agregar `--transcripts-root`.

## Landmarks / MediaPipe guardado

- No se encontraron landmarks guardados bajo `data/processed/lip_rois/` con patrones
  `*landmark*`, `*.json` o `*.npy`.
- Existen los assets de preprocesamiento:
  - `preprocessing/models/face_landmarker.task`
  - `preprocessing/src/20words_mean_face.npy`
- Conclusion: se pueden volver a detectar landmarks con MediaPipe, pero no hay cache de
  landmarks por clip.

## Audio

- Los clips `.mp4` existen bajo `data/clips/`.
- En este entorno local no estan disponibles `ffmpeg` ni `ffprobe`, por lo que no se
  verifico el stream de audio clip por clip.
- Si la VM tiene ffmpeg, validar audio con:

```bash
ffprobe -hide_banner data/clips/<titulo>/<clip>.mp4
```

## Regeneracion de ROIs alternativos

- Es viable conceptualmente usando los clips actuales y el pipeline existente en
  `preprocessing/src/preprocesar.py`.
- Bloqueo local de smoke: `mediapipe` no esta instalado en este entorno.
- Smoke local actual de `lower_face_resized96`: `20/20 ok`, `READY_FOR_FULL_GENERATION`.
- En VM, instalar `preprocessing/requirements.txt` si hiciera falta y correr el
  comando full documentado en `PREPROCESSING_VARIANT.md`.

## Flags actuales de `fine_tune.py`

Verificado en `vsr_models/src/fine_tune.py`:

- `--splits-dir`: soportado; default `vsr_models/splits`.
- `--rois-root`: soportado; requerido.
- `--transcripts-root`: agregado para esta etapa; default vacio, conserva la columna
  `texto` del split.

No se modifican `vsr_models/splits/{train,val,test}.csv`.
