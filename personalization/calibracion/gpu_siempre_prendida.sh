#!/usr/bin/env bash
# Consigue y deja PRENDIDA una L4 fija (labios-cal-gpu) con el daemon de calibracion:
# reintenta spot y on-demand en varias zonas hasta tenerla RUNNING. Con la GPU fija
# arriba, el boton "Entrenar" de la demo dispara directo (sin crear VMs ni esperar).
# Uso: bash personalization/calibracion/gpu_siempre_prendida.sh   (corre hasta lograrlo)
# Apagarla cuando no se usa (factura mientras este prendida):
#   gcloud compute instances stop labios-cal-gpu --zone=<zona>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VM=${VSR_CAL_VM:-labios-cal-gpu}
BUCKET="${VSR_BUCKET:-gs://labios-argentos-vsr-clean-v1}"
PROJ="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[ -n "$PROJ" ] || { echo "Sin proyecto: gcloud config set project <proyecto>"; exit 1; }

sed "s|__BUCKET__|$BUCKET|g" "$HERE/startup_cal_daemon.sh" \
  | gcloud storage cp - "$BUCKET/config/startup_cal_daemon.sh"

while :; do
  INFO=$(gcloud compute instances list --project="$PROJ" --filter="name=$VM" \
         --format="value(zone,status)" 2>/dev/null | head -1)
  case "$INFO" in
    *RUNNING*)
      echo "OK: $VM prendida (zona ${INFO%%	*}). El boton Entrenar ya dispara directo."
      exit 0 ;;
    *TERMINATED*|*SUSPENDED*)
      ZONA=${INFO%%	*}
      echo "  $VM existe apagada: la arranco @ $ZONA"
      gcloud compute instances start "$VM" --zone="$ZONA" --project="$PROJ" >/dev/null 2>&1 || true ;;
    "")
      for MODO in SPOT STANDARD; do
        for ZN in us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d \
                  us-east4-a us-east4-b us-east4-c us-west1-a us-west1-b us-west4-a us-west4-c \
                  europe-west1-b europe-west1-c europe-west4-a europe-west4-b europe-west4-c \
                  asia-southeast1-a asia-southeast1-b asia-northeast1-a; do
          echo "  intento $MODO @ $ZN"
          if [ "$MODO" = SPOT ]; then
            EXTRA="--provisioning-model=SPOT --instance-termination-action=STOP"
          else
            EXTRA="--provisioning-model=STANDARD"
          fi
          if gcloud compute instances create "$VM" --project="$PROJ" --zone="$ZN" \
              --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
              --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
              --image-project=deeplearning-platform-release \
              --boot-disk-type=pd-balanced --boot-disk-size=150GB \
              --maintenance-policy=TERMINATE $EXTRA \
              --metadata=install-nvidia-driver=True,startup-script-url="$BUCKET/config/startup_cal_daemon.sh" \
              --scopes=storage-rw,compute-rw >/dev/null 2>&1; then
            break 2
          fi
        done
      done ;;
  esac
  sleep 10
done
