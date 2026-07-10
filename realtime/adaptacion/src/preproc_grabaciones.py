"""
Recorta el ROI labial de las grabaciones de calibración (frase_NNN.mp4) con el MISMO
pipeline visual del kiosko (`realtime/src/preprocess_live.rois_desde_video`), dejando
pares listos para el fine-tune de adaptación:

    realtime/adaptacion/rois/frase_NNN.npz   # rois (T,96,96) uint8, gris, 25 fps
    realtime/adaptacion/rois/frase_NNN.txt   # transcripción normalizada
    realtime/adaptacion/rois/manifest.csv    # n_frames + ratio de detección por clip

Descarta clips donde la cara no aparece en ≥80% de los cuadros (mismo criterio que el
preproc del dataset). Reanudable: saltea los que ya tienen .npz.

Uso (desde la raíz del repo, env realtime):
    python -m realtime.adaptacion.src.preproc_grabaciones
"""

import csv
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
from realtime.src.preprocess_live import rois_desde_video  # noqa: E402

GRAB = RAIZ / "realtime" / "adaptacion" / "grabaciones"
OUT = RAIZ / "realtime" / "adaptacion" / "rois"
RATIO_MIN = 0.80


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = sorted(GRAB.glob("frase_*.mp4"))
    if not clips:
        print(f"No hay grabaciones en {GRAB}. Grabá primero con grabar_server.py.")
        return

    filas, descartados = [], []
    for i, mp4 in enumerate(clips, 1):
        base = mp4.stem
        npz = OUT / f"{base}.npz"
        if npz.exists():
            continue
        rois, ratio, n_leidos, fps = rois_desde_video(str(mp4))
        ok = rois is not None and len(rois) > 0 and ratio >= RATIO_MIN
        if ok:
            np.savez_compressed(npz, rois=rois)
            (OUT / f"{base}.txt").write_text(
                (GRAB / f"{base}.txt").read_text(encoding="utf-8"), encoding="utf-8")
        else:
            descartados.append(base)
        filas.append({"clip": base, "n_frames": len(rois) if rois is not None else 0,
                      "ratio": round(ratio, 3), "estado": "ok" if ok else "descartado"})
        if i % 20 == 0 or i == len(clips):
            print(f"{i}/{len(clips)} procesados", flush=True)

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "n_frames", "ratio", "estado"])
        w.writeheader()
        w.writerows(filas)

    oks = sum(1 for r in filas if r["estado"] == "ok")
    print(f"\nOK: {oks}/{len(filas)}  ·  descartados: {descartados or 'ninguno'}")
    print(f"ROIs en {OUT}/  (regrabá los descartados con grabar_server.py si querés recuperarlos)")


if __name__ == "__main__":
    main()
