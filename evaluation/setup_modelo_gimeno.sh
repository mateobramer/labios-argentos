#!/usr/bin/env bash
# Setup reproducible del VSR español de Gimeno (modelo del baseline zero-shot rioplatense).
# Deja listo: repo del modelo clonado + checkpoints de Zenodo + entorno conda + nuestros parches.
#
# Los pesos (8.5 GB, 7 checkpoints de ~210 MB) NO se versionan en git: se bajan de Zenodo aquí.
# Probado en la VM labios-vsr-gpu (Ubuntu 22.04 + NVIDIA L4, CUDA 12).
#
# Uso (desde la raíz del repo):  bash evaluation/setup_modelo_gimeno.sh
set -e

REPO_DIR="${REPO_DIR:-$HOME/evaluating-end2end-spanish-lipreading}"
ZEN_DIR="${ZEN_DIR:-$HOME/zenodo}"
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "### 1/4  clonar repo del modelo ###"
[ -d "$REPO_DIR" ] || git clone https://github.com/david-gimeno/evaluating-end2end-spanish-lipreading.git "$REPO_DIR"

echo "### 2/4  checkpoints desde Zenodo (record 17443293, ~8.5 GB) ###"
mkdir -p "$ZEN_DIR"; cd "$ZEN_DIR"
[ -f factors.zip ] || wget -c -O factors.zip \
  "https://zenodo.org/api/records/17443293/files/Factors_of_Influence_on_End-to-End_Continuous_Spanish_Lipreading.zip/content"
echo "c8adb97d4621defc463dae219f4b9be7  factors.zip" | md5sum -c -
[ -d extracted ] || { mkdir -p extracted && unzip -q factors.zip -d extracted; }
echo "    -> checkpoints en $ZEN_DIR/extracted/Factors_*/VSR/  y  /LM/"

echo "### 3/4  entorno conda 'vsr-factors' ###"
# Gotchas resueltos (ver evaluation/README.md para el detalle):
#  - Python 3.8  =>  numpy máx 1.24.4 (1.25+ requiere py3.9).
#  - espnet arrastra deps Cython que no compilan solas -> instalar con --no-build-isolation.
#  - ctc-segmentation pide Cython<3 ; pyworld pide Cython>=3  => se instalan en ese orden.
#  - typeguard==2.13.3: el código de Gimeno usa check_argument_types, removido en typeguard 3.
if [ ! -d "$HOME/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda env list | grep -q vsr-factors || conda create -y -n vsr-factors python=3.8
conda activate vsr-factors
pip install -q torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q "cython<3.0" "numpy==1.24.4" "setuptools<70" wheel pybind11
pip install -q --no-build-isolation "ctc-segmentation==1.7.4"   # necesita Cython<3
pip install -q "cython>=3.0.10"
pip install -q --no-build-isolation pyworld                      # necesita Cython>=3
pip install -q --no-build-isolation espnet colorama pandas unidecode
pip install -q "numpy==1.24.4" "typeguard==2.13.3"

echo "### 4/4  aplicar nuestros parches (registrar base 'Rioplatense') ###"
python "$THIS_DIR/gimeno_patches/aplicar_parches.py" "$REPO_DIR"

cat <<EOF

LISTO. Para inferir (desde $REPO_DIR, con el env vsr-factors activado):

  python vsr_main.py --database Rioplatense --scenario zero-shot \\
    --load-vsr $ZEN_DIR/extracted/Factors_*/VSR/vsr-liprtve-si.pth \\
    --output-dir ./spanish-benchmark/rioplatense/liprtve-si_noLM/

(antes hay que generar los ROIs y exportarlos: ver evaluation/README.md)
EOF
