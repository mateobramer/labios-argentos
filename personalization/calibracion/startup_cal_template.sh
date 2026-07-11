#!/usr/bin/env bash
# Startup L4 (imagen labios-img-visper): CALIBRACION al hablante __PERSONA__ (LoRA, ~5 min).
# SIN eval de test-658 (LoRA ya validado sin olvido en docs/experiments/10) -> run corto y barato.
# Sube el modelo mergeado y SE AUTODESTRUYE (VM + disco boot).
set -uo pipefail
BUCKET=__BUCKET__
HOMEUSER=$(ls /home | head -1)
echo "[$(date -u +%FT%TZ)] CAL startup (user=$HOMEUSER)"
sudo -u "$HOMEUSER" -i bash <<'USERBLOCK'
set -uo pipefail
BUCKET=__BUCKET__
P=__PERSONA__
CF=$BUCKET/calibracion/$P
log(){ echo "[$(date -u +%FT%TZ)] CAL($P): $*"; }
st(){ echo "$*" | gcloud storage cp - $CF/STATUS 2>/dev/null || true; }
source ~/miniconda3/etc/profile.d/conda.sh; conda activate visper
cd ~/visper
pip install -q peft 2>&1 | tail -1
gcloud storage cp $BUCKET/config/fine_tune_visper.py ~/visper/ 2>/dev/null || true
mkdir -p ~/data/cal_$P
gcloud storage cp "$CF/rois/*.npz" ~/data/cal_$P/
gcloud storage cp "$CF/cal_train.csv" "$CF/cal_val.csv" ~/data/
N=$(ls ~/data/cal_$P/*.npz | wc -l); log "npz: $N"; st "SETUP_OK npz=$N"

log "LoRA (r16, lr1e-4, aug, early-stop)"
python fine_tune_visper.py --train-csv ~/data/cal_train.csv --val-csv ~/data/cal_val.csv \
  --data-root ~/data --out ~/cal_$P --lora --lora-r 16 --lora-alpha 32 --augment \
  --lr 1e-4 --accum 8 --epochs 20 --paciencia 5 --max-frames 400 > ~/cal_train.log 2>&1
RC=$?
gcloud storage cp ~/cal_train.log $CF/ 2>/dev/null || true
if [ -f ~/cal_$P/best.pth ]; then
  gcloud storage cp ~/cal_$P/best.pth $CF/${P}.pth
  st "CAL_DONE rc=$RC $(date -u +%FT%TZ)"
  log "===== CAL_DONE ====="
else
  st "CAL_FATAL rc=$RC"; log "CAL_FATAL"
fi
USERBLOCK
echo "[$(date -u +%FT%TZ)] CAL: autodestruyo VM"
NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || true
