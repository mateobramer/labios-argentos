#!/usr/bin/env bash
# Setup de los entornos del proyecto. Idempotente; no toca envs existentes.
# Uso:  bash setup.sh
set -u
cd "$(dirname "$0")"

CONDA_BIN="$(command -v conda || true)"
if [ -z "$CONDA_BIN" ]; then
  for c in ~/miniforge3/bin/conda ~/miniconda3/bin/conda /opt/homebrew/bin/conda; do
    [ -x "$c" ] && CONDA_BIN="$c" && break
  done
fi
if [ -z "$CONDA_BIN" ]; then
  echo "ERROR: no encontré conda (miniforge/miniconda). Instalá miniforge primero." >&2
  exit 1
fi
echo "conda: $CONDA_BIN"

crear_env () {  # $1 = nombre, $2 = yml
  if "$CONDA_BIN" env list | grep -qE "^$1[[:space:]]"; then
    echo "env '$1' ya existe — no se toca."
  else
    echo "creando env '$1' desde $2 ..."
    "$CONDA_BIN" env create -f "$2" || { echo "ERROR creando $1" >&2; exit 1; }
  fi
}

crear_env ptt envs/ptt.yml
crear_env visper envs/visper.yml

# ── artefactos externos ─────────────────────────────────────────────
echo
if [ -f preprocessing/models/face_landmarker.task ]; then
  echo "✓ face_landmarker.task presente (viene versionado en el repo)."
else
  echo "bajando face_landmarker.task de MediaPipe ..."
  curl -L -o preprocessing/models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
fi

VISPER_DIR="${VISPER_DIR:-$HOME/Desktop/visper}"
if [ -f "$VISPER_DIR/visper_vsr_base.pth" ]; then
  echo "✓ pesos ViSpeR encontrados en $VISPER_DIR."
else
  cat <<EOF
✗ FALTA el repo/pesos de ViSpeR (necesarios SOLO para inferencia real):
    1. clonar el repo ViSpeR del equipo en:  $VISPER_DIR   (o exportar VISPER_DIR)
    2. poner visper_vsr_base.pth (1.1 GB) dentro — copia en
       gs://labios-argentos-vsr-dataset o el release oficial de ViSpeR (TII).
  La demo web arranca igual hasta el punto de inferencia; sin esto el
  infer_server falla con import error (comportamiento esperado, sin mocks).
EOF
fi
echo
echo "Opcional (corrector LLM): instalar Ollama y  ollama pull qwen3:4b-instruct-2507-q4_K_M"
echo "Listo. Para correr la demo:  bash run.sh"
