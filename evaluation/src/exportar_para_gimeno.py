"""
Exporta nuestros ROIs labiales al formato que espera el evaluador de VSR de Gimeno
(repo `david-gimeno/evaluating-end2end-spanish-lipreading`), para correr un baseline
zero-shot del modelo español sobre nuestros clips rioplatenses.

Entrada (la produce visual_preprocessing/src/preprocesar.py):
    data/processed/lip_rois/<titulo>/clip_NNNN.npz   # array (T,96,96) uint8, gris, 25 fps
    data/processed/lip_rois/<titulo>/clip_NNNN.txt   # transcripcion limpia (lower+unidecode+ñ)

Salida (layout que arma `src/utils.py:get_dataloader` del repo de Gimeno):
    <salida>/<DB>/ROIs/<spk>/<sampleID>.npz
    <salida>/<DB>/transcriptions/<spk>/<sampleID>.txt
    <salida>/<DB>/splits/<scenario>/test<DB>.csv      # columna "sampleID"
    <salida>/<DB>/mapeo.csv                           # trazabilidad sampleID -> clip original

Convencion de nombres: el loader hace `spk = sampleID[:-delimiter]` (delimiter=5 para la
rama tipo LIP-RTVE). Por eso usamos `sampleID = f"{spk}_{idx:04d}"`: el sufijo "_NNNN"
mide exactamente 5 chars, asi que `spk` (codigo de fuente, p. ej. "f01") sale solo.

Cada fuente (titulo) recibe un codigo de hablante distinto -> escenario speaker-independent.

Uso (desde la raiz del repo):
    python -m evaluation.src.exportar_para_gimeno --max-por-fuente 80 \
        "Story Time CAP 91" "Entrevista por mi libro"
    # sin titulos -> usa todas las fuentes presentes en lip_rois
    # en la VM, escribir directo donde el modelo lo busca (../data desde su repo):
    python -m evaluation.src.exportar_para_gimeno --salida ~/data --db Rioplatense ...
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import cv2  # solo para el fallback de leer mp4 si faltara el npz
except ImportError:
    cv2 = None

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ_REPO = os.path.dirname(DIR_MODULO)
LIP_ROIS_DIR = os.path.join(RAIZ_REPO, "data", "processed", "lip_rois")
SALIDA_DEFAULT = os.path.join(DIR_MODULO, "data", "gimeno_zeroshot")


def cargar_rois(carpeta, base):
    """Devuelve el array (T,96,96) uint8 del clip, desde .npz (preferido) o .mp4 (fallback)."""
    npz_path = os.path.join(carpeta, base + ".npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data[data.files[0]].astype(np.uint8)

    mp4_path = os.path.join(carpeta, base + ".mp4")
    if os.path.exists(mp4_path):
        if cv2 is None:
            raise RuntimeError("falta opencv para leer el mp4 de fallback; reprocesa para tener .npz")
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if f.ndim == 3:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            frames.append(f)
        cap.release()
        return np.asarray(frames, dtype=np.uint8)

    return None


def clips_de_fuente(titulo):
    """Lista ordenada de bases de clip (sin extension) con ROI disponible en una fuente."""
    carpeta = os.path.join(LIP_ROIS_DIR, titulo)
    if not os.path.isdir(carpeta):
        return []
    bases = set()
    for nombre in os.listdir(carpeta):
        raiz, ext = os.path.splitext(nombre)
        if ext in (".npz", ".mp4"):
            bases.add(raiz)
    return sorted(bases)


def main():
    parser = argparse.ArgumentParser(description="Exportar ROIs al formato del evaluador de Gimeno.")
    parser.add_argument("titulos", nargs="*", help="Fuentes a exportar. Vacio = todas las de lip_rois.")
    parser.add_argument("--salida", default=SALIDA_DEFAULT,
                        help="Dir base; se crea <salida>/<DB>/. En la VM conviene '~/data'.")
    parser.add_argument("--db", default="Rioplatense", help="Nombre de la base (DB) para el evaluador.")
    parser.add_argument("--scenario", default="zero-shot", help="Nombre del escenario (subcarpeta de splits).")
    parser.add_argument("--max-por-fuente", type=int, default=0,
                        help="Limite de clips por fuente (0 = sin limite). Util para el subset chico.")
    args = parser.parse_args()
    args.salida = os.path.expanduser(args.salida)

    if not os.path.isdir(LIP_ROIS_DIR):
        print(f"ERROR: no existe '{LIP_ROIS_DIR}'. Corré primero visual_preprocessing.")
        sys.exit(1)

    titulos = args.titulos or sorted(
        d for d in os.listdir(LIP_ROIS_DIR) if os.path.isdir(os.path.join(LIP_ROIS_DIR, d))
    )
    if not titulos:
        print("ERROR: no hay fuentes en lip_rois para exportar.")
        sys.exit(1)

    base_db = os.path.join(args.salida, args.db)
    rois_out = os.path.join(base_db, "ROIs")
    text_out = os.path.join(base_db, "transcriptions")
    splits_out = os.path.join(base_db, "splits", args.scenario)
    for d in (rois_out, text_out, splits_out):
        os.makedirs(d, exist_ok=True)

    sample_ids = []
    mapeo = []  # sampleID, spk, titulo, clip_original, n_frames, texto
    total_clips = total_descartados = 0

    for n, titulo in enumerate(titulos, start=1):
        spk = f"f{n:02d}"
        carpeta = os.path.join(LIP_ROIS_DIR, titulo)
        bases = clips_de_fuente(titulo)
        if args.max_por_fuente > 0:
            bases = bases[: args.max_por_fuente]
        if not bases:
            print(f"  [vacio] {titulo} (no hay ROIs)")
            continue

        os.makedirs(os.path.join(rois_out, spk), exist_ok=True)
        os.makedirs(os.path.join(text_out, spk), exist_ok=True)

        print(f"=== {spk}  {titulo}  ({len(bases)} clips) ===")
        idx = 0
        for base in bases:
            rois = cargar_rois(carpeta, base)
            if rois is None or len(rois) == 0:
                total_descartados += 1
                continue

            txt_path = os.path.join(carpeta, base + ".txt")
            texto = ""
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    texto = f.read().strip()
            if not texto:
                # sin transcripcion no sirve para WER
                total_descartados += 1
                continue

            sample_id = f"{spk}_{idx:04d}"
            np.savez_compressed(os.path.join(rois_out, spk, sample_id + ".npz"), rois=rois)
            with open(os.path.join(text_out, spk, sample_id + ".txt"), "w", encoding="utf-8") as f:
                f.write(texto + "\n")

            sample_ids.append(sample_id)
            mapeo.append([sample_id, spk, titulo, base, len(rois), texto])
            idx += 1
            total_clips += 1

    # -- split CSV (columna "sampleID")
    split_csv = os.path.join(splits_out, f"test{args.db}.csv")
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sampleID"])
        for sid in sample_ids:
            w.writerow([sid])

    # -- mapeo de trazabilidad
    with open(os.path.join(base_db, "mapeo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sampleID", "spk", "titulo", "clip_original", "n_frames", "texto"])
        w.writerows(mapeo)

    print(f"\nListo. {total_clips} clips exportados a '{base_db}' "
          f"({total_descartados} descartados sin ROI/texto).")
    print(f"Split: {split_csv}")
    print(f"El evaluador lo busca en '../data/{args.db}/' relativo a su repo "
          f"(en la VM: ~/data/{args.db}/). Symlinkear/copiar si hace falta.")


if __name__ == "__main__":
    main()
