#!/usr/bin/env bash
# Levanta la demo web con un comando. Flags extra se pasan a demo_web.py
# (p. ej.:  bash run.sh --qwen   |   bash run.sh --ckpt modelos/personal/fede.pth)
set -u
cd "$(dirname "$0")"

PTT_PY=""
for c in ~/miniforge3/envs/ptt/bin/python ~/miniconda3/envs/ptt/bin/python; do
  [ -x "$c" ] && PTT_PY="$c" && break
done
if [ -z "$PTT_PY" ]; then
  echo "ERROR: no encontré el env 'ptt'. Corré primero:  bash setup.sh" >&2
  exit 1
fi
exec "$PTT_PY" demo/demo_web.py "$@"
