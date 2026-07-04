# Comandos VM batch VSR

No crear VM desde este documento. Usar solo cuando la VM/GPU ya exista.

## 0. Verificar entorno

```bash
nvidia-smi
python -V
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
git status --short --branch
```

## 1. Preparar overlays/configs

```bash
python -m evaluation.src.build_visual_cleaning_manifests \
  --splits vsr_models/splits/splits.csv \
  --policy data/metadata/visual_quality_policy_analysis_v2.csv \
  --output-dir evaluation/outputs/visual_cleaning/manifests

python -m evaluation.src.transcript_cleaning \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr

python -m evaluation.src.preprocessing_variant \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --max-clips 2 \
  --preview-max 20

python -m evaluation.src.build_batch_vsr_experiments \
  --output-base evaluation/outputs/batch_vsr
```

Si el smoke queda `blocked_missing_dependency_mediapipe`, instalar:

```bash
pip install -r visual_preprocessing/requirements.txt
```

## 2. Generar preprocessing variant full

Necesario antes de correr E3/E4:

```bash
python -m evaluation.src.preprocessing_variant \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --full \
  --preview-max 20
```

## 3. Entrenamiento con tmux

Crear sesion:

```bash
tmux new -s batch_vsr
```

Comandos por experimento:

```bash
# E0_baseline_original
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root data/processed/lip_rois \
  --splits-dir evaluation/outputs/visual_cleaning/splits_original \
  --out vsr_models/runs/batch_vsr/E0_baseline_original

# E1_visual_cleaned
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root data/processed/lip_rois \
  --splits-dir evaluation/outputs/visual_cleaning/splits_visual_cleaned \
  --out vsr_models/runs/batch_vsr/E1_visual_cleaned

# E2_transcript_cleaned_stronger
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root data/processed/lip_rois \
  --splits-dir evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger \
  --transcripts-root evaluation/outputs/batch_vsr/transcripts_cleaned_stronger \
  --out vsr_models/runs/batch_vsr/E2_transcript_cleaned_stronger

# E3_preprocessing_variant
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root evaluation/outputs/batch_vsr/rois_lower_face_resized96 \
  --splits-dir evaluation/outputs/visual_cleaning/splits_original \
  --out vsr_models/runs/batch_vsr/E3_preprocessing_variant

# E4_all_combined
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root evaluation/outputs/batch_vsr/rois_lower_face_resized96 \
  --splits-dir evaluation/outputs/batch_vsr/splits_all_combined \
  --transcripts-root evaluation/outputs/batch_vsr/transcripts_cleaned_stronger \
  --out vsr_models/runs/batch_vsr/E4_all_combined
```

Detach: `Ctrl-b d`. Reattach: `tmux attach -t batch_vsr`.

## 4. Inferencia y parseo

Despues de evaluar cada `best.pth` con el flujo Gimeno, guardar:

```text
evaluation/outputs/batch_vsr/raw/<experiment>/test.inf
```

Parsear:

```bash
python -m evaluation.src.parse_batch_vsr_results \
  --output-base evaluation/outputs/batch_vsr
```

Salidas:

```text
evaluation/outputs/batch_vsr/results/<experiment>.csv
evaluation/outputs/batch_vsr/results/summary.csv
```

## 5. Notebook

```bash
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('evaluation/notebooks/07_batch_vsr_experiments.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=120, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('evaluation/notebooks/08_transcript_cleaning_review.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=120, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('evaluation/notebooks/09_preprocessing_variant_review.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=120, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
```

## 6. Apagar VM

Desde local o Cloud Shell:

```bash
gcloud compute instances stop labios-vsr-gpu --zone <zona>
```
