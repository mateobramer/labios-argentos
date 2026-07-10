# Preprocessing variant: lower_face_resized96

Objetivo: preparar una variante visual minima compatible con el modelo actual, sin
implementar detectores pesados nuevos.

## Variante propuesta

`lower_face_resized96`:

- entrada: `data/clips/<titulo>/<clip>.mp4`;
- deteccion: MediaPipe existente del modulo `preprocessing`;
- alineacion: `VideoProcess` existente;
- crop: parche mas amplio alrededor de la boca (`128x128`);
- salida final: resize a `96x96`;
- formato: `.npz` con array `rois` de shape `(T, 96, 96)` y dtype `uint8`.

No toca ROIs originales en `data/processed/lip_rois/`.

## Smoke local

Comando:

```bash
python -m preprocessing.src.preprocessing_variant \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --max-clips 20 \
  --preview-max 20
```

Salida:

```text
evaluation/outputs/batch_vsr/preprocessing_variant_smoke/
evaluation/outputs/batch_vsr/preprocessing_variant_manifest_smoke.csv
evaluation/outputs/batch_vsr/preprocessing_variant_preview/
```

Estado local actual: smoke generado con `mediapipe==0.10.35`.

- clips intentados: 20;
- ok: 20;
- failed: 0;
- blocked: 0;
- decision notebook 09: `READY_FOR_FULL_GENERATION`.
- notebook final: `preprocessing/notebooks/09_preprocessing_variant_review.ipynb`.

Las previews locales quedan en:

```text
evaluation/outputs/batch_vsr/preprocessing_variant_preview/
```

Abrir los archivos `*__side_by_side.png` para revisar el ROI actual contra
`lower_face_resized96` antes de generar la variante full.

## Generacion full en VM

Instalar dependencias visuales si hiciera falta:

```bash
pip install -r preprocessing/requirements.txt
```

Generar ROIs alternativos:

```bash
python -m preprocessing.src.preprocessing_variant \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --full \
  --preview-max 20
```

Salida esperada:

```text
evaluation/outputs/batch_vsr/rois_lower_face_resized96/
evaluation/outputs/batch_vsr/preprocessing_variant_manifest_full.csv
```

No commitear `.npz`. Solo versionar scripts, docs y manifests livianos.

Las previews `.png` son para revision visual y estan ignoradas por Git. Se limitan con
`--preview-max`.
