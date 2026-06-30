#!/usr/bin/env bash
# Entrena ft03 (config v1) y ft04 (config v2) sobre old+new, evalua los 4 sobre test 658.
# Corre en la VM creada DESDE el snapshot labios-full (ya tiene repo+venvs+npz nuevos).
# NO se auto-apaga ni snapshotea: el watchdog/Monitor del lado local maneja el cierre.
# Marcadores: PHASE_*_DONE / PHASE_*_FAIL / SMOKE_OK / SMOKE_FAIL / FTxx_DONE / EVAL_* / ALL_DONE / FATAL
set -uo pipefail
cd ~/labios-argentos
log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
GIMENO=$HOME/evaluating-end2end-spanish-lipreading
ROIS=$HOME/labios-argentos/data/processed/lip_rois
VSRCFG=$GIMENO/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml
die(){ log "FATAL: $*"; exit 1; }

log "===== TRAIN ORCHESTRATOR START ====="

# ---- PHASE_SETUP: env Gimeno (conda vsr-factors) + Zenodo + parches ----
if [ ! -f "$HOME/.train_setup_done" ]; then
  # 'unzip' no viene en la imagen base y setup_modelo_gimeno.sh lo necesita (linea 23, descomprime Zenodo).
  command -v unzip >/dev/null || sudo apt-get update -qq >/dev/null 2>&1 && sudo apt-get install -y -qq unzip >/dev/null 2>&1 || true
  log "PHASE_SETUP: setup_modelo_gimeno.sh (puede tardar ~40min: Zenodo 8.5GB)"
  bash evaluation/setup_modelo_gimeno.sh > $HOME/setup_gimeno.log 2>&1 || die "setup_modelo_gimeno fallo (ver setup_gimeno.log)"
  touch $HOME/.train_setup_done
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh" || die "no conda"
conda activate vsr-factors || die "no env vsr-factors"
CKPT=$(ls $HOME/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth 2>/dev/null | head -1)
[ -n "$CKPT" ] || die "no encuentro checkpoint vsr-liprtve-si.pth"
log "PHASE_SETUP_DONE (ckpt=$CKPT)"

# ---- PHASE_DATA: npz viejos + nuevos juntos en $ROIS ----
# OJO: la lectura de gs://labios-argentos-vsr-data/ desde la service account de la VM esta
# ROTA (bucket de otro proyecto, sin IAM). Los npz viejos se suben por scp local->VM (4.2G,
# dataset original). Si ya estan presentes (old+new ~9200), se saltea el rsync.
mkdir -p "$ROIS"
NNPZ=$(find "$ROIS" -name '*.npz' | wc -l)
if [ "$NNPZ" -ge 8000 ]; then
  log "PHASE_DATA: $NNPZ npz ya presentes (old+new) -> salteo rsync GCS"
else
  log "PHASE_DATA: solo $NNPZ npz; intento rsync GCS (fallback)"
  gcloud storage rsync -r gs://labios-argentos-vsr-data/lip_rois "$ROIS" > $HOME/rsync_rois.log 2>&1 || log "rsync GCS fallo (esperado si no hay IAM); sigo con lo presente"
  NNPZ=$(find "$ROIS" -name '*.npz' | wc -l)
fi
[ "$NNPZ" -ge 8000 ] || die "PHASE_DATA: solo $NNPZ npz (<8000) — falta subir los viejos por scp; abortando"
log "PHASE_DATA_DONE (npz totales=$NNPZ)"

# ---- PHASE_SPLITS: armar splits (nuevo->train; val/test fijos) ----
log "PHASE_SPLITS: armar_splits"
python -m vsr_models.src.armar_splits > $HOME/splits.log 2>&1 || die "armar_splits fallo (ver splits.log)"
TR=$(tail -n +2 vsr_models/splits/train.csv 2>/dev/null | wc -l)
VA=$(tail -n +2 vsr_models/splits/val.csv 2>/dev/null | wc -l)
TE=$(tail -n +2 vsr_models/splits/test.csv 2>/dev/null | wc -l)
log "PHASE_SPLITS_DONE (train=$TR val=$VA test=$TE)  [v1/v2 eran train=4818 val=466 test=658]"
[ "$TE" -ge 600 ] || die "test=$TE inesperado (deberia ~658) — abortando para no invalidar la comparacion"
[ "$TR" -gt 4818 ] || log "OJO: train=$TR no aumento respecto a 4818"

FT(){  # $1=run  $2..=flags extra
  local RUN=$1; shift
  python -m vsr_models.src.fine_tune \
    --gimeno-repo "$GIMENO" --vsr-config "$VSRCFG" --load-vsr "$CKPT" \
    --rois-root "$ROIS" --out "vsr_models/runs/$RUN" \
    --lr 1e-4 --batch 1 --accum 8 --max-frames 400 --paciencia 5 "$@"
}

# ---- SMOKE: 1 batch train+val, valida config antes de gastar GPU ----
log "SMOKE: fine_tune --smoke (config v1)"
if FT smoke_test --smoke > $HOME/smoke.log 2>&1; then log "SMOKE_OK"; else log "SMOKE_FAIL (ver smoke.log)"; die "smoke fallo — no entreno"; fi

# ---- FT03 (config v1: full FT) ----
log "FT03 START (config v1: full FT, sin congelar, sin augment)"
FT ft03 > vsr_models/runs/ft03_train.log 2>&1 || die "ft03 entrenamiento fallo"
log "FT03_DONE (best=$(ls -la vsr_models/runs/ft03/best.pth 2>/dev/null | awk '{print $5}'))"

# ---- FT04 (config v2: frontend congelado + augment) ----
log "FT04 START (config v2: --freeze frontend --augment)"
FT ft04 --freeze frontend --augment > vsr_models/runs/ft04_train.log 2>&1 || die "ft04 entrenamiento fallo"
log "FT04_DONE (best=$(ls -la vsr_models/runs/ft04/best.pth 2>/dev/null | awk '{print $5}'))"

# ---- EVAL (best-effort, NO fatal): 4 checkpoints sobre test 658 ----
log "EVAL START (best-effort)"
eval_one(){  # $1=tag  $2=ckpt_path
  local TAG=$1 CK=$2 OUT=$GIMENO/spanish-benchmark/rioplatense/$1
  ( cd "$GIMENO" && python vsr_main.py --database Rioplatense --scenario zero-shot \
      --load-vsr "$CK" --output-dir "$OUT/" ) > $HOME/eval_$TAG.log 2>&1 \
    && log "EVAL_OK $TAG: $(grep -hE '%WER' $OUT/inference/test.wer 2>/dev/null | head -1)" \
    || log "EVAL_FAIL $TAG (ver eval_$TAG.log)"
}
# export del test completo (las 2 fuentes de test, sin cap)
python -m evaluation.src.exportar_para_gimeno --salida $HOME/data --max-por-fuente 9999 \
  "LE DIJE QUE SOY ARGENTINO - Story Time - CAP 91" \
  "ME ACUSARON DE BRUJA Y ME TUVE QUE IR DEL PUEBLO -" > $HOME/export_test.log 2>&1 \
  && log "EVAL export OK" || log "EVAL export FAIL (ver export_test.log)"
# bajar checkpoints v1/v2 de GCS para re-evaluarlos sobre 658
gcloud storage cp gs://labios-argentos-vsr-data/models/ft01_v1/best.pth $HOME/v1_best.pth 2>/dev/null
gcloud storage cp gs://labios-argentos-vsr-data/models/ft02_v2/best.pth $HOME/v2_best.pth 2>/dev/null
eval_one v1   "$HOME/v1_best.pth"
eval_one v2   "$HOME/v2_best.pth"
eval_one ft03 "$HOME/labios-argentos/vsr_models/runs/ft03/best.pth"
eval_one ft04 "$HOME/labios-argentos/vsr_models/runs/ft04/best.pth"
log "EVAL_DONE"

log "===== ALL_DONE ====="
