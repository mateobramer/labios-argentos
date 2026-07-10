# Comandos VM batch VSR

Usar en VM/GPU con `tmux`. No dejar la VM prendida si falla o termina.

## Preflight

```bash
git pull
PYTHONIOENCODING=utf-8 python -m compileall evaluation cleaning/visual_quality preprocessing vsr cleaning/visual_quality/tests
python -m unittest discover cleaning/visual_quality/tests
nvidia-smi
python -V
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Transcript ASR2, si se usa E2

Smoke primero:

```bash
python -m cleaning.visual_quality.src.transcript_second_pass_asr \
  --splits vsr/splits/splits.csv \
  --output vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --model small \
  --max-clips 20
```

Full solo si smoke funciona:

```bash
python -m cleaning.visual_quality.src.transcript_second_pass_asr \
  --splits vsr/splits/splits.csv \
  --output vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --model large-v3-turbo
```

Luego:

```bash
python -m cleaning.visual_quality.src.transcript_alignment_audit \
  --asr2 vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --output vsr/evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv
```

Luego:

```bash
python -m cleaning.visual_quality.src.transcript_cleaning \
  --splits vsr/splits/splits.csv \
  --output-base vsr/evaluation/outputs/batch_vsr \
  --asr2 vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --asr-disagreement vsr/evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv \
  --lexicon cleaning/visual_quality/resources/entity_lexicon.csv
```

## Preprocessing full

```bash
python -m preprocessing.src.preprocessing_variant \
  --splits vsr/splits/splits.csv \
  --output-base vsr/evaluation/outputs/batch_vsr \
  --full \
  --preview-max 20 \
  --workers 3
```

Verificar:

```bash
python - <<'PY'
import re
import pandas as pd
p='vsr/evaluation/outputs/batch_vsr/preprocessing_variant_manifest_full.csv'
df=pd.read_csv(p)
print(df['status'].value_counts(dropna=False))
print(df[['shape', 'dtype']].value_counts(dropna=False).head())
assert (df['status'] == 'ok').all()
assert df['shape'].astype(str).str.match(r'^\d+x96x96$').all()
assert (df['dtype'].astype(str) == 'uint8').all()
PY
```

## Configs

```bash
python -m evaluation.src.build_batch_vsr_experiments \
  --output-base vsr/evaluation/outputs/batch_vsr
```

## Training minimo recomendado

Crear sesion:

```bash
tmux new -s batch_vsr
```

Obligatorios:

```bash
python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root data/processed/lip_rois \
  --splits-dir vsr/evaluation/outputs/visual_cleaning/splits_original \
  --out vsr/runs/batch_vsr/E0_baseline_original

python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root data/processed/lip_rois \
  --splits-dir vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned \
  --out vsr/runs/batch_vsr/E1_visual_cleaned

python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root vsr/evaluation/outputs/batch_vsr/rois_lower_face_resized96 \
  --splits-dir vsr/evaluation/outputs/visual_cleaning/splits_original \
  --out vsr/runs/batch_vsr/E3_preprocessing_variant

python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root vsr/evaluation/outputs/batch_vsr/rois_lower_face_resized96 \
  --splits-dir vsr/evaluation/outputs/batch_vsr/splits_all_combined \
  --transcripts-root vsr/evaluation/outputs/batch_vsr/transcripts_cleaned_stronger \
  --out vsr/runs/batch_vsr/E4_all_combined
```

Opcional, solo si `VM_READINESS.md` marca E2 `READY_FOR_VM`:

```bash
python -m vsr.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \
  --rois-root data/processed/lip_rois \
  --splits-dir vsr/evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger \
  --transcripts-root vsr/evaluation/outputs/batch_vsr/transcripts_cleaned_stronger \
  --out vsr/runs/batch_vsr/E2_transcript_cleaned_stronger
```

Detach: `Ctrl-b d`. Reattach: `tmux attach -t batch_vsr`.

La corrida de verificacion registrada en `VM_READINESS.md` uso:

```bash
EPOCHS=2 BATCH=2 ACCUM=2 TEST_MAX=60 /tmp/run_train_batch.sh
```

`TEST_MAX=60` exporta un subset deterministico balanceado por fuente para que la inferencia
de Gimeno sea finita; no reemplaza una corrida full de convergencia.

## Parseo y notebooks

```bash
python -m evaluation.src.parse_batch_vsr_results \
  --output-base vsr/evaluation/outputs/batch_vsr

python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('vsr/evaluation/notebooks/06_experimentos_cleaning_vs_original.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=180, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('vsr/evaluation/notebooks/07_batch_vsr_experiments.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=180, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('cleaning/visual_quality/notebooks/08_transcript_cleaning_review.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=180, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('preprocessing/notebooks/09_preprocessing_variant_review.ipynb'); nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=180, kernel_name='python3').execute(cwd=str(Path.cwd())); nbformat.write(nb, p)"
```

## Apagar VM

```bash
gcloud compute instances stop labios-vsr-gpu --zone <zona>
```
