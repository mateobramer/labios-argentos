#!/usr/bin/env bash
set -Eeuo pipefail

BUCKET="${BUCKET:-gs://labios-argentos-vsr-clean-v1}"
WORKDIR="${WORKDIR:-/opt/labios-argentos}"
STATUS_JSON="/tmp/vm_run_status.json"
HEARTBEAT="/tmp/vm_heartbeat.txt"
LOG_DIR="/var/log/labios-full-clean"
mkdir -p "$LOG_DIR"

stage="resume"

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
  gcloud storage cp data_pipeline/release/new_discovery_clip_manifest.csv "$BUCKET/argentina/new_discovery/manifests/new_discovery_clip_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/new_discovery_asr_manifest.csv "$BUCKET/argentina/new_discovery/manifests/new_discovery_asr_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/new_discovery_roi_manifest.csv "$BUCKET/argentina/new_discovery/manifests/new_discovery_roi_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/final_release_manifest.csv "$BUCKET/manifests/final_release_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/final_train_manifest_clean_gpt_v1.csv "$BUCKET/manifests/final_train_manifest_clean_gpt_v1.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/final_eval_manifest_clean_gpt_v1.csv "$BUCKET/manifests/final_eval_manifest_clean_gpt_v1.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/clean_gpt_manifest.csv "$BUCKET/manifests/clean_gpt_manifest.csv" >/dev/null 2>&1
  gcloud storage cp data_pipeline/release/reports/* "$BUCKET/reports/" >/dev/null 2>&1
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
  sync_outputs
  exit "$exit_code"
}
trap on_error ERR

heartbeat_loop &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true' EXIT

cd "$WORKDIR"
source .venv-gpu/bin/activate

if [[ -f /tmp/segment_new_discovery_source.py ]]; then
  cp /tmp/segment_new_discovery_source.py data_pipeline/release/scripts/segment_new_discovery_source.py
fi

stage="segment"
write_status "running" "segment_new_discovery_resume"
python data_pipeline/release/scripts/segment_new_discovery_source.py --resume --upload --checkpoint-every 25
sync_outputs

stage="asr"
write_status "running" "asr_large_turbo"
python data_pipeline/release/scripts/transcribe_new_discovery_clips.py \
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
python data_pipeline/release/scripts/generate_new_discovery_rois.py --resume --upload --checkpoint-every 25
sync_outputs

stage="build"
write_status "running" "build_manifests"
python data_pipeline/release/scripts/build_full_clean_release_outputs.py
sync_outputs

stage="validate"
write_status "running" "validate_bucket"
python data_pipeline/release/scripts/validate_clean_bucket.py
gcloud storage cp data_pipeline/release/reports/bucket_validation_report.md "$BUCKET/reports/bucket_validation_report.md"
sync_outputs

stage="done"
write_status "completed" "processing_complete"
