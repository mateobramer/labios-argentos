# Comandos VM para visual cleaning

Estos comandos preparan la comparacion, pero no deben ejecutarse en local si requieren
GPU. Ajustar rutas de VM antes de correr.

## 1. Generar manifests locales

```bash
python -m evaluation.src.build_visual_cleaning_manifests \
  --splits vsr/splits/splits.csv \
  --policy data/metadata/visual_quality_policy_analysis_v2.csv \
  --output-dir vsr/evaluation/outputs/visual_cleaning/manifests
```

Esto tambien genera splits compatibles con `fine_tune.py`:

```text
vsr/evaluation/outputs/visual_cleaning/splits_original/train.csv
vsr/evaluation/outputs/visual_cleaning/splits_original/val.csv
vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned/train.csv
vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned/val.csv
```

`splits_visual_cleaned/val.csv` conserva la val original completa. La decision es
conservadora: el experimento filtra solo train y mantiene val comparable.

## 2. baseline_original

Entrena con los splits originales copiados a un directorio de experimento. No pisa
`vsr/splits/`.

```bash
python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root ~/data/lip_rois \
  --splits-dir vsr/evaluation/outputs/visual_cleaning/splits_original \
  --out vsr/runs/baseline_original
```

Evaluar WER/CER con el flujo del repo de Gimeno sobre el test original completo. El
script de training imprime que el `best.pth` debe evaluarse con `vsr_main.py`; conservar
el `.inf` y `.wer` de esa evaluacion.

## 3. visual_cleaned_conservative

Entrena excluyendo solo `training_usability == bad_candidate` del train. Son 203 de
4826 clips (~4.2%), por lo que el efecto global esperado puede ser chico.

```bash
python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root ~/data/lip_rois \
  --splits-dir vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned \
  --out vsr/runs/visual_cleaned_conservative
```

El test principal sigue siendo el test original completo. No generar ni usar un test
filtrado como numero titular.

## 4. Inferencia y outputs esperados

Guardar resultados parseables bajo:

```text
vsr/evaluation/outputs/visual_cleaning/results/baseline_original_test.csv
vsr/evaluation/outputs/visual_cleaning/results/visual_cleaned_test_original.csv
```

Formato esperado en `results_schema.md`.

Si se conserva salida cruda de Gimeno:

```text
vsr/evaluation/outputs/visual_cleaning/raw/baseline_original/test.inf
vsr/evaluation/outputs/visual_cleaning/raw/baseline_original/test.wer
vsr/evaluation/outputs/visual_cleaning/raw/visual_cleaned/test.inf
vsr/evaluation/outputs/visual_cleaning/raw/visual_cleaned/test.wer
```

## 5. Traer outputs desde VM

Ejemplo con `gcloud compute scp`:

```bash
gcloud compute scp --recurse labios-vsr-gpu:~/labios-argentos/evaluation/outputs/visual_cleaning \
  vsr/evaluation/outputs/
```

Verificar que los CSV finales tengan el test original completo antes de comparar.

## 6. Parseo de resultados

El repositorio ya tiene parseos especificos para salidas `.inf` en la etapa visual
(`cleaning/visual_quality/src/visual_quality_vsr_results.py`). Para esta comparacion
especifica (dataset original vs cleaning visual, ver `EXPERIMENT_PLAN.md`) no se llego a
escribir un parser general que convirtiera `.inf` + manifest full-test al schema estandar
de `results_schema.md`; el resultado final de esta linea exploratoria no se cerro.
