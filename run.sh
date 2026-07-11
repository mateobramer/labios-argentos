#!/usr/bin/env bash
# Levanta la demo web con un comando. Flags extra se pasan a demo_web.py
# (p. ej.:  bash run.sh --qwen   |   bash run.sh --ckpt modelos/personal/fede.pth)
set -u
cd "$(dirname "$0")"

# Config por máquina (gitignoreada): exporta lo definido en .env si existe. Ver .env.example.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Corrector LLM: si Ollama está instalado pero apagado, levantarlo (la UI avisa si falta).
if ! curl -s --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  OLLAMA_BIN="$(command -v ollama || true)"
  [ -z "$OLLAMA_BIN" ] && [ -x "$HOME/.local/ollama/ollama" ] && OLLAMA_BIN="$HOME/.local/ollama/ollama"
  if [ -n "$OLLAMA_BIN" ]; then
    echo "[run] levantando Ollama para el corrector..."
    mkdir -p "$HOME/.ollama"
    nohup "$OLLAMA_BIN" serve >> "$HOME/.ollama/serve.log" 2>&1 &
  fi
fi

find_env_python() {
  local env_name="$1"
  local explicit_path="${2:-}"

  if [ -n "$explicit_path" ]; then
    if [ -x "$explicit_path" ]; then
      printf '%s\n' "$explicit_path"
      return 0
    fi
    echo "ERROR: el Python configurado para '$env_name' no existe o no es ejecutable: $explicit_path" >&2
    return 1
  fi

  local candidate
  for candidate in \
    "$HOME/miniforge3/envs/$env_name/bin/python" \
    "$HOME/miniconda3/envs/$env_name/bin/python"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local conda_bin
  conda_bin="$(command -v conda || true)"
  if [ -z "$conda_bin" ]; then
    for candidate in \
      "$HOME/miniforge3/bin/conda" \
      "$HOME/miniconda3/bin/conda" \
      "/opt/homebrew/bin/conda"; do
      if [ -x "$candidate" ]; then
        conda_bin="$candidate"
        break
      fi
    done
  fi

  if [ -n "$conda_bin" ]; then
    local env_path
    env_path="$("$conda_bin" env list 2>/dev/null | awk -v name="$env_name" '$1 == name {print $NF; exit}')"
    if [ -n "$env_path" ] && [ -x "$env_path/bin/python" ]; then
      printf '%s\n' "$env_path/bin/python"
      return 0
    fi
  fi

  return 1
}

PTT_PY="$(find_env_python ptt "${PTT_PY:-}")" || {
  echo "ERROR: no encontré el env 'ptt'. Corré primero: bash setup.sh" >&2
  exit 1
}

VISPER_PY_RESOLVED="$(find_env_python visper "${VISPER_PY:-}")" || {
  echo "ERROR: no encontré el env 'visper'. Corré primero: bash setup.sh" >&2
  exit 1
}

export VISPER_PY="$VISPER_PY_RESOLVED"
exec "$PTT_PY" demo/demo_web.py "$@"
