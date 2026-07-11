#!/usr/bin/env bash
# Calibracion al hablante, de punta a punta (despues de grabar en /calibrar):
#   1. arma splits de ~/vsr_personal/<persona>/
#   2. sube ROIs+CSVs al bucket
#   3. levanta una L4 spot (retry multi-zona) que entrena LoRA (~5 min) y se autodestruye
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
VM=labios-cal-$P

echo "== 1/4 splits =="
"${VSR_CAL_PY:-python3}" "$HERE/armar_splits_cal.py" "$P"

echo "== 2/4 subida =="
gcloud storage cp "$SRC"/clip_*.npz "$CF/rois/" | tail -1
gcloud storage cp "$SRC/cal_train.csv" "$SRC/cal_val.csv" "$CF/"
gcloud storage rm "$CF/STATUS" 2>/dev/null || true
sed -e "s/__PERSONA__/$P/g" -e "s|__BUCKET__|$BUCKET|g" "$HERE/startup_cal_template.sh" \
  | gcloud storage cp - "$BUCKET/config/startup_cal_$P.sh"

echo "== 3/4 VM L4 (spot, retry multi-zona) =="
# Cuota tipica del proyecto: 1 GPU total. Si otra VM la ocupa, avisar claro en vez
# de recorrer zonas como si fuera falta de capacidad de Google.
CUOTA=$(gcloud compute project-info describe --format=json 2>/dev/null | python3 -c '
import json,sys
for q in json.load(sys.stdin).get("quotas", []):
    if q["metric"] == "GPUS_ALL_REGIONS":
        print(int(q["usage"]), int(q["limit"])); break' 2>/dev/null || true)
if [ -n "$CUOTA" ]; then
  set -- $CUOTA
  if [ "$1" -ge "$2" ]; then
    echo "CUOTA DE GPU AGOTADA: $1/$2 GPUs del proyecto en uso. VMs corriendo:"
    gcloud compute instances list --filter="status=RUNNING"       --format="table(name,zone,machineType.basename())" 2>/dev/null
    echo "Apagá o borrá la VM que usa la GPU (o pedí mas cuota) y volvé a intentar."
    exit 1
  fi
fi
ERRLOG=$(mktemp)
LANZADA=""
for Z in us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-west1-a; do
  echo "  intento spot @ $Z"
  if gcloud compute instances create "$VM" --project=$PROJ --zone=$Z \
      --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
      --image=labios-img-visper --boot-disk-type=pd-balanced \
      --maintenance-policy=TERMINATE \
      --provisioning-model=SPOT --instance-termination-action=DELETE \
      --metadata=startup-script-url="$BUCKET/config/startup_cal_$P.sh" \
      --scopes=storage-rw,compute-rw >/dev/null 2>>"$ERRLOG"; then
    LANZADA=$Z; break
  fi
done
if [ -z "$LANZADA" ]; then
  echo "SIN CAPACIDAD L4 spot; pruebo L4 on-demand"
  for Z in us-central1-a us-central1-b us-east1-b; do
    echo "  intento on-demand @ $Z"
    if gcloud compute instances create "$VM" --project=$PROJ --zone=$Z \
      --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
      --image=labios-img-visper --boot-disk-type=pd-balanced \
      --maintenance-policy=TERMINATE \
      --metadata=startup-script-url="$BUCKET/config/startup_cal_$P.sh" \
      --scopes=storage-rw,compute-rw >/dev/null 2>>"$ERRLOG"; then LANZADA=$Z; break; fi
  done
fi
[ -z "$LANZADA" ] && { echo "NO SE PUDO CREAR LA VM en ninguna zona. Ultimo error real de gcloud:"; grep -v '^$' "$ERRLOG" | tail -4; exit 1; }
echo "  lanzada en $LANZADA"

echo "== 4/4 esperando (~8-12 min; entrena y se autodestruye) =="
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
gcloud storage cp "$CF/$P.pth" "$REPO/modelos/personal/$P.pth"
echo ""
echo "LISTO ✅  modelo personal: modelos/personal/$P.pth"
echo "Para usarlo:  ~/miniconda3/envs/ptt/bin/python $REPO/demo/demo_web.py --ckpt modelos/personal/$P.pth"
