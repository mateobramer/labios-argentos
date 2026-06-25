"""
Arma los splits train/val/test del fine-tuning, **speaker-independent**.

Lee la curaduria (`data/metadata/auditoria_clips_manifest.csv`, estado=keep) y asigna
cada clip a un split SEGUN SU FUENTE (ninguna fuente cae en dos splits), para no inflar
las metricas con el mismo hablante en train y test.

Salida (formato **framework-agnostico**, lo consume cualquier arquitectura):
    vsr_models/splits/splits.csv            # split, spk, titulo, clip, n_frames, texto, npz
    vsr_models/splits/{train,val,test}.csv  # idem, filtrado por split

Cada fila apunta al ROI en `data/processed/lip_rois/<titulo>/<clip>.npz` (array (T,96,96)
uint8 gris a 25 fps) + su transcripcion limpia. Asi un companero con OTRA arquitectura
(o una capa agentica) consume el MISMO dataset y los MISMOS splits, sin atarse a ESPnet.

Uso (desde la raiz del repo):
    python -m vsr_models.src.armar_splits
"""

import csv
import os
from collections import Counter

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ_REPO = os.path.dirname(DIR_MODULO)
MANIFEST = os.path.join(RAIZ_REPO, "data", "metadata", "auditoria_clips_manifest.csv")
SALIDA_DIR = os.path.join(DIR_MODULO, "splits")
ROIS_REL = os.path.join("data", "processed", "lip_rois")

# Splits speaker-independent: se eligen FUENTES ENTERAS para val y test; el resto va a
# train. Test = las fuentes del baseline zero-shot, para comparar fine-tuned vs zero-shot
# sobre el mismo material. Cambiar estas listas y re-correr re-arma los splits.
# (Nombres exactos del manifest; estan truncados a 50 chars por nombre_carpeta.)
TEST_FUENTES = {
    "LE DIJE QUE SOY ARGENTINO - Story Time - CAP 91",
    "ME ACUSARON DE BRUJA Y ME TUVE QUE IR DEL PUEBLO -",
}
VAL_FUENTES = {
    "AZZARO REACCIÓN - CICLO TERMINADO RACING EMPATÓ 2-",
    "CHARLA SOBRE EL AMOR Y EL DESAMOR",
    "PROFESIONES ARGENTINAS GINECÓLOGOS - Telefe Notici",
}


def split_de(titulo):
    if titulo in TEST_FUENTES:
        return "test"
    if titulo in VAL_FUENTES:
        return "val"
    return "train"


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["estado"] == "keep"]
    fuentes = sorted({r["titulo"] for r in rows})

    # Guard: que cada fuente elegida para val/test exista (atrapa typos/truncados).
    faltan = (TEST_FUENTES | VAL_FUENTES) - set(fuentes)
    if faltan:
        raise SystemExit(f"ERROR: estas fuentes de val/test no existen en el manifest:\n  "
                         + "\n  ".join(sorted(faltan)))

    # Codigo de hablante estable por fuente (orden alfabetico) -> f01, f02, ...
    spk = {t: f"f{i:02d}" for i, t in enumerate(fuentes, start=1)}

    filas = []
    for r in rows:
        t = r["titulo"]
        filas.append({
            "split": split_de(t), "spk": spk[t], "titulo": t, "clip": r["clip"],
            "n_frames": r["n_frames"], "texto": r["texto"],
            "npz": os.path.join(ROIS_REL, t, r["clip"] + ".npz"),
        })
    filas.sort(key=lambda x: (x["split"], x["titulo"], x["clip"]))

    os.makedirs(SALIDA_DIR, exist_ok=True)
    cols = ["split", "spk", "titulo", "clip", "n_frames", "texto", "npz"]

    def escribir(path, datos):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(datos)

    escribir(os.path.join(SALIDA_DIR, "splits.csv"), filas)
    for s in ("train", "val", "test"):
        datos = [x for x in filas if x["split"] == s]
        escribir(os.path.join(SALIDA_DIR, f"{s}.csv"), datos)
        nspk = len({x["titulo"] for x in datos})
        print(f"{s:5s}: {len(datos):5d} clips  ({nspk} hablantes)")

    # Chequeo duro: ninguna fuente cae en dos splits.
    por_fuente = {}
    for x in filas:
        por_fuente.setdefault(x["titulo"], set()).add(x["split"])
    fugas = {t: s for t, s in por_fuente.items() if len(s) > 1}
    assert not fugas, f"FUGA speaker-independent: {fugas}"

    print(f"\nTotal: {len(filas)} clips, {len(fuentes)} hablantes. Speaker-independent OK.")
    print(f"Splits en: {SALIDA_DIR}")


if __name__ == "__main__":
    main()
