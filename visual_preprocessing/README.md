# visual_preprocessing

Preparacion visual de los clips para VSR.

Este modulo vive despues de `descargar_procesar.py`: toma los clips alineados en
`data/clips/` y genera recortes labiales listos para revisar o entrenar.

## Que va aca

- deteccion de rostro;
- landmarks faciales;
- recorte de boca/labios;
- normalizacion visual basica: fps, tamano y escala de grises;
- filtros de calidad visual;
- previews para revisar a ojo los clips candidatos a descartar.

La limpieza textual o de alineacion video-texto vive en `data_cleaning/`.

## Estructura

```text
visual_preprocessing/
  README.md
  requirements.txt
  models/
    face_landmarker.task
  outputs/
    previews/
  src/
    auditar_calidad_visual.py
    preprocesar.py
    vista_previa.py
```

## Uso

Instalar dependencias en un entorno separado si MediaPipe choca con Whisper/Torch:

```bash
pip install -r visual_preprocessing/requirements.txt
```

Procesar todos los clips pendientes:

```bash
python -m visual_preprocessing.src.preprocesar
```

Procesar una fuente puntual:

```bash
python -m visual_preprocessing.src.preprocesar "LE DIJE QUE SOY ARGENTINO - Story Time - CAP 91"
```

Salida:

```text
data/processed/lip_rois/<titulo>/clip_NNNN.mp4
data/processed/lip_rois/<titulo>/clip_NNNN.txt
data/metadata/lip_preprocessing_manifest.csv
```

El manifest registra tambien clips descartados por baja deteccion de cara. No borra
clips originales.

## Notebooks

Crear notebooks solo cuando haya una revision concreta con codigo y resultados. No
dejamos notebooks vacios como plan.

La revision de calidad visual del dataset se trabaja desde:

```text
data_cleaning/notebooks/01_revision_visual_mediapipe.ipynb
```

Ese notebook importa funciones de `visual_preprocessing/src/auditar_calidad_visual.py`.

## Pendiente importante

Antes de procesar todo el dataset como version final, confirmar contra el loader de
entrenamiento si Auto-AVSR espera `.mp4`, tensores `.npy`, otra normalizacion, o una
alineacion facial distinta.
