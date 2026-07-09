#!/usr/bin/env bash
# Adaptación al hablante: fine-tune CONTINUADO desde el modelo base sobre las frases de
# una persona, y eval en su test personal (antes/después). Correr EN LA VM (env
# vsr-factors, con el repo de Gimeno y vsr_models/src/fine_tune.py disponibles).
#
# Requiere en el bucket del proyecto:
#   - modelo base:            $BASE_CKPT           (ej. models/ft06/best.pth)
#   - ROIs personales:        $AD/rois/*.npz       (subidos desde realtime/adaptacion/rois/)
#   - splits + export Gimeno: $AD/kit/...          (subidos desde realtime/adaptacion/splits + Personal/)
# Ajustá BUCKET/paths a tu proyecto. Sube el adapter y su WER; NO apaga la VM solo
# (agregá `sudo poweroff` al final si querés que se apague).
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vsr-factors

BUCKET=gs://labios-argentos-vsr-data
AD=$BUCKET/adaptacion
BASE_CKPT=$BUCKET/models/ft06/best.pth      # modelo campeón general a personalizar
TAG=${1:-adapt_persona}                       # nombre del adapter
GIM=~/evaluating-end2end-spanish-lipreading
CFG=$GIM/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml

cd ~
mkdir -p ~/data/adapt_rois/joaco ~/subsets
gcloud storage cp "$BASE_CKPT" ~/base.pth
gcloud storage cp "$AD/rois/*.npz" ~/data/adapt_rois/joaco/
gcloud storage cp "$AD/adapt_subsets/*.csv" ~/subsets/            # train/val/test personales

# test personal como escenario de la base Rioplatense (lo que espera vsr_main.py)
R=~/data/Rioplatense
mkdir -p "$R/ROIs/p01" "$R/transcriptions/p01" "$R/splits/personal-test"
gcloud storage cp "$AD/kit/adaptacion_kit/Personal/ROIs/p01/*"            "$R/ROIs/p01/"
gcloud storage cp "$AD/kit/adaptacion_kit/Personal/transcriptions/p01/*"  "$R/transcriptions/p01/"
gcloud storage cp "$AD/kit/adaptacion_kit/Personal/splits/personal-test/testPersonal.csv" \
                  "$R/splits/personal-test/testRioplatense.csv"

sed -i "s/beam_size: .*/beam_size: 10/;s/lm_weight: .*/lm_weight: 0.0/" "$CFG"
evalp () { cd "$GIM"
  python vsr_main.py --database Rioplatense --scenario personal-test \
    --load-vsr "$2" --output-dir "./out_eval/$1/" >/dev/null 2>&1
  echo "$1 -> $(cat "./out_eval/$1/inference/test.wer" 2>/dev/null | head -1)"; cd ~; }

echo "### WER personal ANTES (base sin adaptar) ###"
evalp base_persona ~/base.pth

echo "### fine-tune de adaptación ###"
cp ~/subsets/train.csv ~/subsets/val.csv ~/subsets/test.csv ~/labios-argentos/vsr_models/splits/
cd ~/labios-argentos
python -u -m vsr_models.src.fine_tune --gimeno-repo "$GIM" --vsr-config "$CFG" \
  --load-vsr ~/base.pth --rois-root ~/data/adapt_rois --out vsr_models/runs/$TAG \
  --epochs 20 --batch 4 --accum 1 --max-frames 400 --lr 1e-5 --paciencia 4 --augment

CKPT=~/labios-argentos/vsr_models/runs/$TAG/best.pth
echo "### WER personal DESPUÉS (adaptado) ###"
evalp $TAG "$CKPT"

gcloud storage cp "$CKPT" "$AD/${TAG}_best.pth"
echo "adapter subido a $AD/${TAG}_best.pth"
