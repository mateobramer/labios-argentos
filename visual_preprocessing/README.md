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

## Como funciona (alineacion a cara media, estilo Auto-AVSR)

Por cada cuadro del clip:

1. **MediaPipe FaceLandmarker** detecta 478 landmarks faciales.
2. Se extraen **4 puntos estables**: ojo derecho, ojo izquierdo, punta de nariz y
   centro de boca.
3. **`video_process.py`** (adaptado de Auto-AVSR) estima una **transformacion afin**
   que alinea esos 4 puntos contra una **cara media de referencia**
   (`20words_mean_face.npy`), deja la cara en pose/escala/rotacion canonicas, pasa a
   gris y recorta **96x96** centrado en la boca. Suaviza los puntos en el tiempo e
   interpola cuadros sin deteccion.
4. Se descartan los clips con cara detectada en menos de `UMBRAL_DETECCION` (80%) de
   los cuadros.

Esto es lo que el modelo de Auto-AVSR espera: a diferencia de un recorte por
bounding-box recto, el warp deja la boca siempre en la misma posicion, escala y
orientacion, sin importar como se mueva la cabeza del hablante.

> El recorte 96x96 en disco coincide con el formato de Auto-AVSR. El modelo, al cargar,
> aplica ademas `CenterCrop(88)` + `Normalize(mean=0.421, std=0.165)`; eso se hace en el
> entrenamiento, no aca.

## Estructura

```text
visual_preprocessing/
  README.md
  requirements.txt
  models/
    face_landmarker.task        # modelo de MediaPipe (no se versiona; bajar con curl)
  outputs/
    previews/
  src/
    preprocesar.py              # pipeline: detecta -> alinea (warp) -> recorta
    video_process.py            # warp a cara media (adaptado de Auto-AVSR, Apache-2.0)
    20words_mean_face.npy       # cara de referencia para la alineacion
    vista_previa.py             # hoja de contactos para revisar a ojo
    auditar_calidad_visual.py
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

- **Alineacion a cara media: hecha** (warp afin estilo Auto-AVSR, ver "Como funciona").
- Falta confirmar contra el loader de entrenamiento si Auto-AVSR espera el clip como
  `.mp4` o como tensores `.npy`, y si los puntos estables de MediaPipe (ojos/nariz/boca)
  dan una alineacion equivalente a la del detector original de 68 puntos (RetinaFace).
  Si hace falta paridad exacta, se puede usar el `video_process.py` con el detector
  RetinaFace de Auto-AVSR en lugar de MediaPipe.
