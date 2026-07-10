#!/usr/bin/env bash
# Fase A: procesa las 39 fuentes (descarga + large-v2 + clips + filtro musica + preproc + deteccion).
# NO se auto-apaga: deja la VM viva al terminar para que se puedan bajar los resultados por scp.
# El tope de costo lo garantiza --max-run-duration de la VM.
set -uo pipefail
cd ~/labios-argentos
log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "=== SETUP START ==="
sudo apt-get update -qq >/dev/null 2>&1
sudo apt-get install -y -qq ffmpeg python3.10-venv python3-dev \
  libgl1 libglib2.0-0 libgles2 libegl1 libegl-mesa0 >/dev/null 2>&1
log "apt OK (ffmpeg=$(command -v ffmpeg||echo NO) libGLESv2=$(ldconfig -p|grep -c libGLESv2))"
python3 -m venv --system-site-packages ~/venv-proc
~/venv-proc/bin/pip -q install --upgrade pip >/dev/null 2>&1
~/venv-proc/bin/pip -q install -r requirements.txt >/dev/null 2>&1
python3 -m venv ~/venv-visual
~/venv-visual/bin/pip -q install --upgrade pip >/dev/null 2>&1
~/venv-visual/bin/pip -q install -r visual_preprocessing/requirements.txt >/dev/null 2>&1
~/venv-proc/bin/python -c "import whisper,yt_dlp,torch;assert torch.cuda.is_available()" || { log "FATAL env proc/cuda"; exit 1; }
~/venv-visual/bin/python -c "import mediapipe,cv2" || { log "FATAL env visual"; exit 1; }
log "=== SETUP DONE (proc+visual OK, cuda OK) ==="

PROC=~/venv-proc/bin/python; VIS=~/venv-visual/bin/python
export WHISPER_MODEL=large-v2
mapfile -t URLS < <(python3 -c "import csv;[print(r['url']) for r in csv.DictReader(open('candidatos.csv'))]")
N=${#URLS[@]}
log "=== PIPELINE START: $N fuentes | modelo $WHISPER_MODEL ==="
i=0
for URL in "${URLS[@]}"; do
  i=$((i+1))
  log "[$i/$N] >>> descargando+transcribiendo $URL"
  OUT=$($PROC descargar_procesar.py "$URL" 2>&1); rc=$?
  TIT=$(echo "$OUT" | grep '^Carpeta:' | tail -1 | sed 's/^Carpeta: //')
  if [ $rc -ne 0 ] || [ -z "$TIT" ]; then
    log "[ERROR $i/$N] fallo descarga/transcripcion: $URL"; echo "$OUT" | tail -4; continue
  fi
  NC=$(ls "data/clips/$TIT/"*.mp4 2>/dev/null | wc -l | tr -d ' ')
  MUS=$($PROC filtro_musica.py "$TIT" 2>/dev/null | grep '^music_dropped=' | cut -d= -f2)
  NC2=$(ls "data/clips/$TIT/"*.mp4 2>/dev/null | wc -l | tr -d ' ')
  if ! $VIS -m visual_preprocessing.src.preprocesar "$TIT" --jobs 6 >>~/preproc.log 2>&1; then
    log "[WARN $i/$N] preproc fallo en $TIT (ver preproc.log)"
  fi
  NN=$(ls "data/processed/lip_rois/$TIT/"*.npz 2>/dev/null | wc -l | tr -d ' ')
  log "[DONE $i/$N] $TIT | clips=$NC music_drop=${MUS:-0} -> $NC2 | npz=$NN"
done

log "=== DETECCION final keep/review/drop ==="
$VIS -m data_cleaning.src.detectar_clips_malos >>~/preproc.log 2>&1 || log "WARN deteccion"

log "=== RESUMEN ==="
$PROC - <<'PY'
import os
r="data/processed/lip_rois"; c="data/clips"
tc=tn=0
print("%-52s %7s %7s %6s" % ("FUENTE","clips","npz","pass%"))
for t in (sorted(os.listdir(r)) if os.path.isdir(r) else []):
    nn=len([x for x in os.listdir(f"{r}/{t}") if x.endswith(".npz")])
    nc=len([x for x in os.listdir(f"{c}/{t}") if x.endswith(".mp4")]) if os.path.isdir(f"{c}/{t}") else 0
    tc+=nc; tn+=nn
    print("%-52s %7d %7d %5.1f%%" % (t[:52],nc,nn,100*nn/nc if nc else 0))
print("%-52s %7d %7d %5.1f%%" % ("TOTAL",tc,tn,100*tn/tc if tc else 0))
PY
log "=== PIPELINE_DONE (VM sigue viva para bajar resultados; tope max-run-duration) ==="
