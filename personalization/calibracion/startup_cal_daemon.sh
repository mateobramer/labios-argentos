#!/usr/bin/env bash
# Startup de la GPU FIJA (labios-cal-gpu, imagen labios-img-visper): queda esperando
# pedidos de calibracion en el bucket (calibracion/<persona>/GO) y entrena cada uno
# con la misma receta LoRA validada (docs/experiments/10). NO se autodestruye: la
# prende/mantiene gpu_siempre_prendida.sh y la apaga el equipo cuando no hace falta.
# Corre en cada boot (tambien tras una preempcion de spot + start).
set -uo pipefail
BUCKET=__BUCKET__
HOMEUSER=$(ls /home | head -1)
echo "[$(date -u +%FT%TZ)] CAL daemon startup (user=$HOMEUSER)"
sudo -u "$HOMEUSER" -i bash <<'USERBLOCK'
set -uo pipefail
BUCKET=__BUCKET__
source ~/miniconda3/etc/profile.d/conda.sh; conda activate visper
cd ~/visper
pip install -q peft 2>&1 | tail -1
gcloud storage cp $BUCKET/config/fine_tune_visper.py ~/visper/ 2>/dev/null || true
echo "[$(date -u +%FT%TZ)] daemon listo: espero GO en $BUCKET/calibracion/*/GO"
while :; do
  for GO in $(gcloud storage ls "$BUCKET/calibracion/*/GO" 2>/dev/null); do
    P=$(basename "$(dirname "$GO")")
    CF=$BUCKET/calibracion/$P
    log(){ echo "[$(date -u +%FT%TZ)] CAL($P): $*"; }
    st(){ echo "$*" | gcloud storage cp - $CF/STATUS 2>/dev/null || true; }
    gcloud storage rm "$GO" 2>/dev/null || true
    log "pedido recibido"
    mkdir -p ~/data/cal_$P
    gcloud storage cp "$CF/rois/*.npz" ~/data/cal_$P/ || { st "CAL_FATAL sin_rois"; continue; }
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
  done
  sleep 20
done
USERBLOCK
