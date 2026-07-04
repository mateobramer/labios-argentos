# Preprocessing variant: lower_face_resized96

Objetivo: preparar una variante visual minima compatible con el modelo actual, sin
implementar detectores pesados nuevos.

## Variante propuesta

`lower_face_resized96`:

- entrada: `data/clips/<titulo>/<clip>.mp4`;
- deteccion: MediaPipe existente del modulo `visual_preprocessing`;
- alineacion: `VideoProcess` existente;
- crop: parche mas amplio alrededor de la boca (`128x128`);
- salida final: resize a `96x96`;
- formato: `.npz` con array `rois` de shape `(T, 96, 96)` y dtype `uint8`.

No toca ROIs originales en `data/processed/lip_rois/`.

## Smoke local

Comando:

```bash
python -m evaluation.src.preprocessing_variant \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --max-clips 2
```

Salida:

```text
evaluation/outputs/batch_vsr/preprocessing_variant_smoke/
evaluation/outputs/batch_vsr/preprocessing_variant_manifest_smoke.csv
```

Estado local actual: `blocked_missing_dependency_mediapipe`. En este entorno `cv2` y
`numpy` estan instalados, pero `mediapipe` no. El asset
`visual_preprocessing/models/face_landmarker.task` si existe.

## Generacion full en VM

Instalar dependencias visuales si hiciera falta:

```bash
pip install -r visual_preprocessing/requirements.txt
```

Generar ROIs alternativos:

```bash
python -m evaluation.src.preprocessing_variant \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --full
```

Salida esperada:

```text
evaluation/outputs/batch_vsr/rois_lower_face_resized96/
evaluation/outputs/batch_vsr/preprocessing_variant_manifest_full.csv
```

No commitear `.npz`. Solo versionar scripts, docs y manifests livianos.
