#!/usr/bin/env bash
# Startup de la GPU FIJA (labios-cal-gpu) sobre imagen PUBLICA de Deep Learning VM
# (la imagen custom labios-img-visper fue borrada). Se auto-provisiona desde el bucket:
#   config/visper_code.tgz        codigo vsr/visper del repo (lo sube la Mac)
#   config/fine_tune_visper.py    receta LoRA (la sube el equipo)
#   config/visper_vsr_base.pth    pesos base 1.1GB (los sube el equipo)
# Si faltan piezas, ESPERA y avisa via config/DAEMON_STATUS; cuando aparecen, sigue.
# Luego atiende pedidos: calibracion/<persona>/GO -> entrena LoRA -> sube delta.
# NO se autodestruye: la mantiene gpu_siempre_prendida.sh.
set -uo pipefail
BUCKET=__BUCKET__
echo "[$(date -u +%FT%TZ)] CAL daemon startup"
sudo -i bash <<'ROOTBLOCK'
set -uo pipefail
BUCKET=__BUCKET__
dst(){ echo "[$(date -u +%FT%TZ)] $*" | gcloud storage cp - $BUCKET/config/DAEMON_STATUS 2>/dev/null || true; }
cd /root

# --- provision (idempotente; corre en cada boot) ---
if [ ! -d visper ]; then
  gcloud storage cp $BUCKET/config/visper_code.tgz . && tar xzf visper_code.tgz && rm -f visper_code.tgz
fi
python3 -m pip install -q peft pytorch-lightning sentencepiece 2>&1 | tail -1
while [ ! -f visper/visper_vsr_base.pth ]; do
  if gcloud storage cp $BUCKET/config/visper_vsr_base.pth visper/ 2>/dev/null; then break; fi
  dst "ESPERANDO visper_vsr_base.pth en $BUCKET/config/ (subir con: gcloud storage cp ~/visper/visper_vsr_base.pth $BUCKET/config/)"
  sleep 60
done
while [ ! -f visper/fine_tune_visper.py ]; do
  if gcloud storage cp $BUCKET/config/fine_tune_visper.py visper/ 2>/dev/null; then break; fi
  dst "ESPERANDO fine_tune_visper.py en $BUCKET/config/"
  sleep 60
done
dst "DAEMON LISTO: GPU esperando pedidos GO"
cd /root/visper

# --- loop de pedidos ---
while :; do
  for GO in $(gcloud storage ls "$BUCKET/calibracion/*/GO" 2>/dev/null); do
    P=$(basename "$(dirname "$GO")")
    CF=$BUCKET/calibracion/$P
    log(){ echo "[$(date -u +%FT%TZ)] CAL($P): $*"; }
    st(){ echo "$*" | gcloud storage cp - $CF/STATUS 2>/dev/null || true; }
    gcloud storage rm "$GO" 2>/dev/null || true
    log "pedido recibido"
    gcloud storage cp $BUCKET/config/fine_tune_visper.py . 2>/dev/null || true   # version fresca
    mkdir -p /root/data/cal_$P
    gcloud storage cp "$CF/rois/*.npz" /root/data/cal_$P/ || { st "CAL_FATAL sin_rois"; continue; }
    gcloud storage cp "$CF/cal_train.csv" "$CF/cal_val.csv" /root/data/
    N=$(ls /root/data/cal_$P/*.npz | wc -l); log "npz: $N"; st "SETUP_OK npz=$N"
    log "LoRA (r16, lr1e-4, aug, early-stop)"
    python3 fine_tune_visper.py --train-csv /root/data/cal_train.csv --val-csv /root/data/cal_val.csv \
      --data-root /root/data --out /root/cal_$P --lora --lora-r 16 --lora-alpha 32 --augment \
      --lr 1e-4 --accum 8 --epochs 20 --paciencia 5 --max-frames 400 > /root/cal_train.log 2>&1
    RC=$?
    gcloud storage cp /root/cal_train.log $CF/ 2>/dev/null || true
    if [ -f /root/cal_$P/best.pth ]; then
      # subir SOLO el delta vs base: la bajada en la Mac pasa de 1.1 GB a decenas de MB
      python3 - "$P" <<'PYDELTA'
import sys, torch
p = sys.argv[1]
sd = lambda d: d.get("state_dict", d) if isinstance(d, dict) else d
base = sd(torch.load("/root/visper/visper_vsr_base.pth", map_location="cpu"))
best = sd(torch.load(f"/root/cal_{p}/best.pth", map_location="cpu"))
delta = {k: v for k, v in best.items() if k not in base or not torch.equal(base[k], v)}
torch.save(delta, f"/root/cal_{p}/delta.pth")
print(f"[delta] {len(delta)}/{len(best)} tensores cambiaron")
PYDELTA
      if [ -f /root/cal_$P/delta.pth ]; then
        gcloud storage cp /root/cal_$P/delta.pth $CF/${P}_delta.pth
      else
        gcloud storage cp /root/cal_$P/best.pth $CF/${P}.pth   # fallback: modelo completo
      fi
      st "CAL_DONE rc=$RC $(date -u +%FT%TZ)"
      log "===== CAL_DONE ====="
    else
      st "CAL_FATAL rc=$RC"; log "CAL_FATAL"
    fi
  done
  sleep 20
done
ROOTBLOCK
