"""
Arma los splits personales para el fine-tune de adaptación, a partir de los ROIs de
`realtime/adaptacion/rois/`. Deja dos cosas:

  1. splits/{train,val,test}.csv  -> formato que consume `vsr_models/src/fine_tune.py`
     (columnas split,spk,titulo,clip,n_frames,texto,npz; titulo="joaco", spk="p01").
  2. Personal/  -> export en formato del evaluador de Gimeno, para medir el WER personal
     sobre el test retenido (ROIs/transcriptions/splits + mapeo.csv).

Split determinístico (por defecto ~96/9/15): test = cada 8vo clip, val = cada 11vo de los
restantes, train = el resto. Cambiá TEST_CADA / VAL_CADA para otras proporciones.

Uso (desde la raíz del repo):
    python -m realtime.adaptacion.src.armar_splits_personal
"""

import csv
import shutil
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
ROIS = RAIZ / "realtime" / "adaptacion" / "rois"
BASE = RAIZ / "realtime" / "adaptacion"
SPLITS = BASE / "splits"
DB = BASE / "Personal"

SPK, TITULO = "p01", "joaco"       # código de hablante / nombre de fuente para la persona
TEST_CADA, VAL_CADA = 8, 11        # 1 de cada N va a test / val
COLS = ["split", "spk", "titulo", "clip", "n_frames", "texto", "npz"]


def fila(clip, split):
    rois = np.load(ROIS / f"{clip}.npz")["rois"]
    texto = (ROIS / f"{clip}.txt").read_text(encoding="utf-8").strip()
    return {"split": split, "spk": SPK, "titulo": TITULO, "clip": clip,
            "n_frames": len(rois), "texto": texto,
            "npz": f"realtime/adaptacion/rois/{clip}.npz"}


def main():
    clips = sorted(p.stem for p in ROIS.glob("frase_*.npz"))
    if not clips:
        print(f"No hay ROIs en {ROIS}. Corré preproc_grabaciones.py primero.")
        return
    test = set(clips[TEST_CADA - 1::TEST_CADA])
    resto = [c for c in clips if c not in test]
    val = set(resto[VAL_CADA - 1::VAL_CADA])
    train = [c for c in resto if c not in val]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    filas = ([fila(c, "train") for c in train]
             + [fila(c, "val") for c in sorted(val)]
             + [fila(c, "test") for c in sorted(test)])
    SPLITS.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        with open(SPLITS / f"{s}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
            w.writerows([x for x in filas if x["split"] == s])

    # export Gimeno del test personal (para vsr_main.py --scenario personal-test)
    rois_out, txt_out = DB / "ROIs" / SPK, DB / "transcriptions" / SPK
    split_out = DB / "splits" / "personal-test"
    for d in (rois_out, txt_out, split_out):
        d.mkdir(parents=True, exist_ok=True)
    ids = []
    for j, c in enumerate(sorted(test)):
        sid = f"{SPK}_{j:04d}"
        shutil.copy2(ROIS / f"{c}.npz", rois_out / f"{sid}.npz")
        shutil.copy2(ROIS / f"{c}.txt", txt_out / f"{sid}.txt")
        ids.append((sid, c))
    with open(split_out / "testPersonal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sampleID"])
        for sid, _ in ids:
            w.writerow([sid])
    with open(DB / "mapeo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sampleID", "clip"]); w.writerows(ids)

    print(f"splits en {SPLITS}/  ·  export de test en {DB}/")


if __name__ == "__main__":
    main()
