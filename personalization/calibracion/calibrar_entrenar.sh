#!/usr/bin/env bash
# Calibracion al hablante, de punta a punta (despues de grabar en /calibrar):
#   1. arma splits de ~/vsr_personal/<persona>/
#   2. sube ROIs+CSVs al bucket
#   3. dispara el entrenamiento (GO) en la GPU FIJA labios-cal-gpu; si esta apagada,
#      gpu_siempre_prendida.sh la consigue primero (reintenta hasta tenerla)
#   4. espera y baja el modelo a modelos/personal/<persona>.pth
# Costo tipico: ~$0.05. Uso:  bash personalization/calibracion/calibrar_entrenar.sh <persona>
set -euo pipefail
P=${1:?uso: calibrar_entrenar.sh <persona>}
P=$(echo "$P" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SRC="${VSR_PERSONAL_DIR:-$HOME/vsr_personal}/$P"
BUCKET="${VSR_BUCKET:-gs://labios-argentos-vsr-clean-v1}"
CF=$BUCKET/calibracion/$P
PROJ="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [ -z "$PROJ" ] || [ "$PROJ" = "(unset)" ]; then
  echo "No hay proyecto activo: configurá GCP_PROJECT o corré gcloud config set project <proyecto>."
  exit 1
fi

echo "== 1/4 splits =="
"${VSR_CAL_PY:-python3}" "$HERE/armar_splits_cal.py" "$P"

echo "== 2/4 subida (rsync: solo lo que no este ya arriba) =="
gcloud storage rsync "$SRC" "$CF/rois/" --exclude=".*\.(txt|csv|json|tmp)$" | tail -1
gcloud storage cp "$SRC/cal_train.csv" "$SRC/cal_val.csv" "$CF/"
gcloud storage rm "$CF/STATUS" 2>/dev/null || true
echo "== 3/4 GPU fija (labios-cal-gpu) =="
# Modelo nuevo: una L4 FIJA con el daemon de calibracion corre siempre; entrenar es
# dejar un archivo GO en el bucket. Si la GPU no esta prendida, gpu_siempre_prendida.sh
# la consigue (reintenta hasta tenerla) y recien entonces se dispara el GO.
VM_FIJA=${VSR_CAL_VM:-labios-cal-gpu}
ESTADO=$(gcloud compute instances list --project="$PROJ" --filter="name=$VM_FIJA" \
         --format="value(status)" 2>/dev/null | head -1)
if [ "$ESTADO" != "RUNNING" ]; then
  echo "  $VM_FIJA no esta prendida (${ESTADO:-no existe}): la consigo ahora (puede tardar)..."
  bash "$HERE/gpu_siempre_prendida.sh" || { echo "No pude conseguir la GPU fija."; exit 1; }
  echo "  dandole 90 s para bootear el daemon..."
  sleep 90
fi
echo "  disparo el entrenamiento (GO) para $P"
gcloud storage rm "$CF/STATUS" 2>/dev/null || true
echo "$(date -u +%FT%TZ)" | gcloud storage cp - "$CF/GO"

echo "== 4/4 esperando (~8-12 min en la GPU fija) =="
for i in $(seq 1 60); do
  sleep 30
  ST=$(gcloud storage cat "$CF/STATUS" 2>/dev/null || echo "...")
  echo "  [$((i*30/60)) min] $ST"
  case "$ST" in
    CAL_DONE*)  break ;;
    CAL_FATAL*) echo "FALLO el entrenamiento — log:"; gcloud storage cat "$CF/cal_train.log" | tail -20; exit 1 ;;
  esac
done

mkdir -p "$REPO/modelos/personal"
if gcloud storage cp "$CF/${P}_delta.pth" "$REPO/modelos/personal/${P}_delta.pth" 2>/dev/null; then
  "${VSR_CAL_PY:-python3}" "$HERE/aplicar_delta.py" \
    "$REPO/modelos/personal/${P}_delta.pth" "$REPO/modelos/personal/$P.pth"
  rm -f "$REPO/modelos/personal/${P}_delta.pth"
else
  gcloud storage cp "$CF/$P.pth" "$REPO/modelos/personal/$P.pth"   # fallback: completo
fi
echo ""
echo "LISTO ✅  modelo personal: modelos/personal/$P.pth"
echo "Para usarlo:  ~/miniconda3/envs/ptt/bin/python $REPO/demo/demo_web.py --ckpt modelos/personal/$P.pth"
