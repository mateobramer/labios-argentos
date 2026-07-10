#!/usr/bin/env bash
# Plan 2: intenta agarrar una GPU (L4 luego T4, us-central1 a/b/c) y, si lo logra,
# sube el codigo + 39 URLs + filtro de musica y lanza el pipeline de Fase A en la VM.
# Imprime LANZADO / SIN_CAPACIDAD / YA_EXISTE. Idempotente.
set -uo pipefail
REPO=/Users/fedegutman/Desktop/labios-argentos
SCR=$REPO/new-data-fine-tuning/scripts
cd "$REPO"
VM=labios-vsr-full
CSV=$REPO/claude-videos/candidatos.csv
ARCH=/tmp/code-lean-autograb.tgz
IMG="--image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release --boot-disk-size=100GB --boot-disk-type=pd-balanced --scopes=https://www.googleapis.com/auth/cloud-platform"
DUR="--maintenance-policy=TERMINATE --max-run-duration=43200s --instance-termination-action=STOP"

# si ya existe la VM (corrida previa), no duplicar
EX=$(gcloud compute instances list --filter="name=$VM" --format="value(name,zone,status)" 2>/dev/null)
[ -n "$EX" ] && { echo "YA_EXISTE: $EX"; exit 0; }

git archive --format=tar.gz -o "$ARCH" HEAD -- \
  descargar_procesar.py requirements.txt visual_preprocessing data_cleaning vsr_models evaluation data/metadata 2>/dev/null

get_vm(){
  for Z in us-central1-a us-central1-b us-central1-c; do
    O=$(gcloud compute instances create $VM --zone=$Z --machine-type=g2-standard-8 $DUR $IMG --quiet 2>&1)
    echo "$O" | grep -q RUNNING && { echo "$Z|L4"; return 0; }
  done
  for Z in us-central1-a us-central1-b us-central1-c; do
    O=$(gcloud compute instances create $VM --zone=$Z --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 $DUR $IMG --quiet 2>&1)
    echo "$O" | grep -q RUNNING && { echo "$Z|T4"; return 0; }
  done
  return 1
}

GOT=""
for r in $(seq 1 8); do
  echo "ronda $r @ $(date -u +%H:%M:%SZ)"
  if GOT=$(get_vm); then echo "GOT=$GOT"; break; fi
  GOT=""; sleep 180
done
[ -z "$GOT" ] && { echo "SIN_CAPACIDAD @ $(date -u +%FT%TZ)"; exit 0; }

ZONE=${GOT%%|*}; TYPE=${GOT#*|}
SSH(){ gcloud compute ssh $VM --zone=$ZONE --quiet --command="$1" 2>/dev/null; }
n=0; until SSH "echo ok" | grep -q ok; do n=$((n+1)); [ $n -ge 24 ] && break; sleep 10; done
SSH "mkdir -p ~/labios-argentos"
gcloud compute scp "$ARCH" $VM:~/code-lean.tgz --zone=$ZONE --quiet 2>/dev/null
SSH "tar xzf ~/code-lean.tgz -C ~/labios-argentos"
gcloud compute scp "$CSV" $VM:~/labios-argentos/candidatos.csv --zone=$ZONE --quiet 2>/dev/null
gcloud compute scp "$SCR/filtro_musica.py" $VM:~/labios-argentos/filtro_musica.py --zone=$ZONE --quiet 2>/dev/null
gcloud compute scp "$SCR/setup_and_run.sh" $VM:~/setup_and_run.sh --zone=$ZONE --quiet 2>/dev/null
SSH "rm -f ~/run.log; nohup bash ~/setup_and_run.sh > ~/run.log 2>&1 & echo PID \$!"
echo "LANZADO zone=$ZONE type=$TYPE STARTEPOCH=$(date +%s) VM=$VM @ $(date -u +%FT%TZ)"
