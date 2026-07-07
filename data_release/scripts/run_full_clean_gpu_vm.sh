#!/usr/bin/env bash
set -Eeuo pipefail

BUCKET="${BUCKET:-gs://labios-argentos-vsr-clean-v1}"
REPO_URL="${REPO_URL:-https://github.com/mateobramer/labios-argentos.git}"
BRANCH="${BRANCH:-feature/full-clean-release}"
WORKDIR="${WORKDIR:-/opt/labios-argentos}"
STATUS_JSON="/tmp/vm_run_status.json"
HEARTBEAT="/tmp/vm_heartbeat.txt"
LOG_DIR="/var/log/labios-full-clean"
mkdir -p "$LOG_DIR"

stage="boot"

write_status() {
  local status="$1"
  local reason="${2:-}"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "$STATUS_JSON" <<EOF
{"status":"$status","stage":"$stage","reason":"$reason","updated_at":"$now","host":"$(hostname)"}
EOF
  gcloud storage cp "$STATUS_JSON" "$BUCKET/reports/vm_run_status.json" >/dev/null 2>&1 || true
}

sync_outputs() {
  set +e
  cd "$WORKDIR" 2>/dev/null || return 0
  gcloud storage cp data_release/new_discovery_clip_manifest.csv "$BUCKET/argentina/new_discovery/manifests/new_discovery_clip_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_release/new_discovery_asr_manifest.csv "$BUCKET/argentina/new_discovery/manifests/new_discovery_asr_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_release/new_discovery_roi_manifest.csv "$BUCKET/argentina/new_discovery/manifests/new_discovery_roi_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_release/final_release_manifest.csv "$BUCKET/manifests/final_release_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_release/final_train_manifest_clean_gpt_v1.csv "$BUCKET/manifests/final_train_manifest_clean_gpt_v1.csv" >/dev/null 2>&1
  gcloud storage cp data_release/final_eval_manifest_clean_gpt_v1.csv "$BUCKET/manifests/final_eval_manifest_clean_gpt_v1.csv" >/dev/null 2>&1
  gcloud storage cp data_release/clean_gpt_manifest.csv "$BUCKET/manifests/clean_gpt_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_release/reports/*.md "$BUCKET/reports/" >/dev/null 2>&1
  set -e
}

best_effort_closeout() {
  set +e
  cd "$WORKDIR" 2>/dev/null || return 0
  if [[ -x ".venv-gpu/bin/python" ]]; then
    .venv-gpu/bin/python data_release/scripts/build_full_clean_release_outputs.py
    .venv-gpu/bin/python data_release/scripts/validate_clean_bucket.py
  fi
  set -e
}

heartbeat_loop() {
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$HEARTBEAT"
    printf 'stage=%s host=%s\n' "$stage" "$(hostname)" >> "$HEARTBEAT"
    gcloud storage cp "$HEARTBEAT" "$BUCKET/reports/vm_heartbeat.txt" >/dev/null 2>&1 || true
    sleep 60
  done
}

on_error() {
  local exit_code=$?
  write_status "failed" "exit_code=$exit_code"
  best_effort_closeout
  sync_outputs
  exit "$exit_code"
}
trap on_error ERR

heartbeat_loop &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true' EXIT

write_status "running" "startup"

stage="install"
write_status "running" "installing_dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git ffmpeg python3-venv python3-pip

stage="checkout"
write_status "running" "checkout_repo"
if [[ -d "$WORKDIR/.git" ]]; then
  cd "$WORKDIR"
  git fetch origin
else
  rm -rf "$WORKDIR"
  git clone "$REPO_URL" "$WORKDIR"
  cd "$WORKDIR"
fi
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

stage="python_env"
write_status "running" "installing_python_dependencies"
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -r visual_preprocessing/requirements.txt

stage="preflight"
write_status "running" "gpu_preflight"
nvidia-smi | tee "$LOG_DIR/nvidia-smi.txt"
python - <<'PY'
import importlib.util
mods = ["faster_whisper", "mediapipe", "cv2", "numpy"]
for mod in mods:
    print(mod, bool(importlib.util.find_spec(mod)))
PY

stage="sync_manifests"
write_status "running" "syncing_manifests_from_gcs"
mkdir -p data_release/reports data_release/human_review_pack
gcloud storage cp "$BUCKET/reports/local_source_download_manifest.csv" data_release/local_source_download_manifest.csv || true
gcloud storage cp "$BUCKET/argentina/new_discovery/manifests/new_discovery_clip_manifest.csv" data_release/new_discovery_clip_manifest.csv || true
gcloud storage cp "$BUCKET/argentina/new_discovery/manifests/new_discovery_asr_manifest.csv" data_release/new_discovery_asr_manifest.csv || true
gcloud storage cp "$BUCKET/argentina/new_discovery/manifests/new_discovery_roi_manifest.csv" data_release/new_discovery_roi_manifest.csv || true

stage="segment"
write_status "running" "segment_new_discovery"
python data_release/scripts/segment_new_discovery_source.py --resume --upload --checkpoint-every 25
sync_outputs

stage="asr"
write_status "running" "asr_large_turbo"
python data_release/scripts/transcribe_new_discovery_clips.py \
  --model large=large-v3 \
  --model turbo=turbo \
  --device cuda \
  --compute-type float16 \
  --beam-size 1 \
  --checkpoint-every 25 \
  --resume \
  --upload
sync_outputs

stage="roi"
write_status "running" "roi_mediapipe"
python data_release/scripts/generate_new_discovery_rois.py --resume --upload --checkpoint-every 25
sync_outputs

stage="build"
write_status "running" "build_manifests"
python data_release/scripts/build_full_clean_release_outputs.py
sync_outputs

stage="validate"
write_status "running" "validate_bucket"
python data_release/scripts/validate_clean_bucket.py
gcloud storage cp data_release/reports/bucket_validation_report.md "$BUCKET/reports/bucket_validation_report.md"

stage="done"
sync_outputs
write_status "completed" "processing_complete"
