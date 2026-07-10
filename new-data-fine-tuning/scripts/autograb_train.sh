#!/usr/bin/env bash
# Auto-grab de GPU (L4 luego T4, us-central1 a/b/c). Al conseguir, crea VM DESDE el snapshot
# de Fase A (trae repo+venvs+npz nuevos) y lanza train_orchestrator.sh. Idempotente.
set -uo pipefail
SCR=/Users/fedegutman/Desktop/labios-argentos/new-data-fine-tuning/scripts
VM=labios-vsr-train
SNAP=labios-full-20260629-0429
DISKOPT="--create-disk=boot=yes,source-snapshot=$SNAP,size=100,type=pd-balanced"
COMMON="--maintenance-policy=TERMINATE --max-run-duration=43200s --instance-termination-action=STOP --scopes=https://www.googleapis.com/auth/cloud-platform"

EX=$(gcloud compute instances list --filter="name=$VM" --format="value(name)" 2>/dev/null)
[ -n "$EX" ] && { echo "YA_EXISTE: $VM"; exit 0; }

get_vm(){
  for Z in us-central1-a us-central1-b us-central1-c; do
    O=$(gcloud compute instances create $VM --zone=$Z --machine-type=g2-standard-8 $DISKOPT $COMMON --quiet 2>&1)
    echo "$O" | grep -q RUNNING && { echo "$Z|L4"; return 0; }
  done
  for Z in us-central1-a us-central1-b us-central1-c; do
    O=$(gcloud compute instances create $VM --zone=$Z --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 $DISKOPT $COMMON --quiet 2>&1)
    echo "$O" | grep -q RUNNING && { echo "$Z|T4"; return 0; }
  done
  return 1
}

GOT=""
for r in $(seq 1 8); do
  echo "ronda $r @ $(date -u +%H:%M:%SZ)"
  if GOT=$(get_vm); then echo "GOT=$GOT"; break; fi
  GOT=""; echo "  sin capacidad L4+T4; espero 180s"; sleep 180
done
[ -z "$GOT" ] && { echo "SIN_CAPACIDAD @ $(date -u +%FT%TZ)"; exit 0; }

ZONE=${GOT%%|*}; TYPE=${GOT#*|}
SSH(){ gcloud compute ssh $VM --zone=$ZONE --quiet --ssh-flag="-o ConnectTimeout=20" --command="$1" 2>/dev/null; }
n=0; until SSH "echo ok" | grep -q ok; do n=$((n+1)); [ $n -ge 30 ] && break; sleep 12; done
gcloud compute scp "$SCR/train_orchestrator.sh" $VM:~/train_orchestrator.sh --zone=$ZONE --quiet 2>/dev/null
SSH "rm -f ~/train.log; nohup bash ~/train_orchestrator.sh > ~/train.log 2>&1 & echo PID \$!"
echo "LANZADO_TRAIN zone=$ZONE type=$TYPE VM=$VM STARTEPOCH=$(date +%s) @ $(date -u +%FT%TZ)"
