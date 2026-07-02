# Comandos VM para visual cleaning

Estos comandos preparan la comparacion, pero no deben ejecutarse en local si requieren
GPU. Ajustar rutas de VM antes de correr.

## 1. Generar manifests locales

```bash
python -m evaluation.src.build_visual_cleaning_manifests \
  --splits vsr_models/splits/splits.csv \
  --policy data/metadata/visual_quality_policy_analysis_v2.csv \
  --output-dir evaluation/outputs/visual_cleaning/manifests
```

## 2. baseline_original

`vsr_models/src/fine_tune.py` ya entrena con `vsr_models/splits/train.csv` y
`vsr_models/splits/val.csv`.

```bash
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root ~/data/lip_rois \
  --out vsr_models/runs/baseline_original
```

Evaluar WER/CER con el flujo del repo de Gimeno sobre el test original completo. El
script de training imprime que el `best.pth` debe evaluarse con `vsr_main.py`; conservar
el `.inf` y `.wer` de esa evaluacion.

## 3. visual_cleaned

Gap actual: `vsr_models/src/fine_tune.py` hardcodea `vsr_models/splits/{train,val}.csv`
y no acepta `--splits-dir` ni manifests custom. No sobrescribir los splits canonicos del
repo.

Opciones seguras antes de entrenar en VM:

1. Agregar una bandera pequena `--splits-dir` al training loop y apuntarla a
   `evaluation/outputs/visual_cleaning/manifests/`.
2. Usar un worktree/copia temporal de VM y copiar ahi:
   - `visual_cleaned_train.csv` como `vsr_models/splits/train.csv`;
   - `visual_cleaned_val.csv` como `vsr_models/splits/val.csv`;
   - `visual_cleaned_test_original.csv` como `vsr_models/splits/test.csv`.

No hacer esa copia en el repo de trabajo principal.

Placeholder una vez resuelto el gap:

```bash
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root ~/data/lip_rois \
  --out vsr_models/runs/visual_cleaned
```

## 4. Inferencia y outputs esperados

Guardar resultados parseables bajo:

```text
evaluation/outputs/visual_cleaning/results/baseline_original_test.csv
evaluation/outputs/visual_cleaning/results/visual_cleaned_test_original.csv
```

Formato esperado en `results_schema.md`.

Si se conserva salida cruda de Gimeno:

```text
evaluation/outputs/visual_cleaning/raw/baseline_original/test.inf
evaluation/outputs/visual_cleaning/raw/baseline_original/test.wer
evaluation/outputs/visual_cleaning/raw/visual_cleaned/test.inf
evaluation/outputs/visual_cleaning/raw/visual_cleaned/test.wer
```

## 5. Traer outputs desde VM

Ejemplo con `gcloud compute scp`:

```bash
gcloud compute scp --recurse labios-vsr-gpu:~/labios-argentos/evaluation/outputs/visual_cleaning \
  evaluation/outputs/
```

Verificar que los CSV finales tengan el test original completo antes de comparar.

## 6. Parseo de resultados

El repositorio ya tiene parseos especificos para salidas `.inf` en la etapa visual
(`data_cleaning/src/visual_quality_vsr_results.py`). Para esta comparacion falta un
parser general comprometido que convierta `.inf` + manifest full-test al schema estandar
de `results_schema.md`.

Gap documentado: crear o extender ese parser antes de cerrar resultados finales, usando
`source_id`, `clip` y `reference` del manifest de test original.
