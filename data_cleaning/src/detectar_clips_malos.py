"""
Detector de clips malos para VSR, sobre los ROIs labiales 96x96.

Corre DESPUES del preprocesamiento visual: lee los recortes alineados en
`data/processed/lip_rois/<titulo>/clip_NNNN.mp4` (lo que efectivamente ve el modelo)
y marca clips inservibles para fine-tuning. No usa MediaPipe: trabaja directo sobre
los pixeles del ROI, asi que es barato (solo cv2 + numpy).

Tres familias de problemas (los que pidio el proyecto):

1. NEGRO / OSCURO  -> frames casi negros (placa, fundido, camara tapada).
2. CONGELADO       -> imagen practicamente estatica (sin habla real / freeze).
3. BOCA INACTIVA   -> la region de la boca casi no cambia en el tiempo y tiene poca
                      textura pese a que el clip tiene transcripcion. Es un proxy
                      *heuristico* de boca tapada (mano/microfono/mate), mala
                      alineacion o silencio. Por eso se marca como `review`, no `drop`.

Salida:
    data/metadata/auditoria_clips_manifest.csv   (un unico manifest, estados keep/review/drop)

Uso desde la raiz del repo (con cv2 + numpy instalados):
    python -m data_cleaning.src.detectar_clips_malos                 # audita todo
    python -m data_cleaning.src.detectar_clips_malos "<titulo>"      # una fuente
    python -m data_cleaning.src.detectar_clips_malos --materializar  # copia los keep -> dataset/

El paso `--materializar` arma el dataset final curado en `dataset/` copiando solo los
clips con estado `keep` (mp4 + txt) y deja un `dataset/manifest.csv`. No borra nada de
`data/processed/lip_rois/`: ese es el insumo crudo y se conserva.
"""

import csv
import os
import shutil
import sys

import cv2
import numpy as np


# --- rutas -------------------------------------------------------------------
RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROIS_DIR = os.path.join(RAIZ_REPO, "data", "processed", "lip_rois")
MANIFEST_PATH = os.path.join(RAIZ_REPO, "data", "metadata", "auditoria_clips_manifest.csv")
DATASET_DIR = os.path.join(RAIZ_REPO, "dataset")
DATASET_MANIFEST = os.path.join(DATASET_DIR, "manifest.csv")

# --- umbrales (escala 0-255) -------------------------------------------------
# Calibrados sobre las 9 fuentes ya procesadas (1704 ROIs). Percentiles observados:
#   luma_media        p1=72.8  p50=112  (no hay clips negros: minimo 64)
#   movimiento_global p1=2.65  p50=6.1  (no hay congelados: minimo 1.9)
#   actividad_boca    p1=8.3   p50=18   textura_boca p1=15.7 p5=18.5 p50=74
# Los umbrales de DROP son red de seguridad para FUTURAS fuentes (hoy no disparan);
# los de REVIEW se sitúan en el piso real para surfacear la cola dudosa a revision.

# Negro / oscuro -> DROP
UMBRAL_LUMA_FRAME_NEGRA = 25.0   # un frame con luma media < esto cuenta como "oscuro"
UMBRAL_LUMA_CLIP = 45.0          # si la luma media del clip entero cae por debajo -> negro
UMBRAL_FRAC_OSCUROS = 0.5        # fraccion de frames oscuros que dispara "negro"

# Congelado (movimiento global) -> DROP
UMBRAL_MOV_GLOBAL = 1.2          # diff temporal media (0-255) por debajo -> imagen estatica

# Boca inactiva / oclusion (region central del ROI 96x96) -> REVIEW (señal blanda)
CAJA_BOCA = (28, 68, 18, 78)     # (fila0, fila1, col0, col1) centrada en la boca
UMBRAL_ACTIVIDAD_BOCA = 9.0      # std temporal media de la boca; tapada/quieta -> baja
UMBRAL_TEXTURA_BOCA = 18.0       # varianza de Laplaciano media; piel/mano lisa -> baja
MIN_PALABRAS_TEXTO = 2           # solo exigimos actividad de boca si hay habla esperada

ENCABEZADO = [
    "titulo", "clip", "n_frames",
    "luma_media", "frac_oscuros", "movimiento_global",
    "actividad_boca", "textura_boca",
    "estado", "razones", "texto",
]


def leer_frames_gris(clip_path):
    """Lee el clip como un array [T, H, W] float32 en escala de grises."""
    cap = cv2.VideoCapture(clip_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


def metricas_clip(frames):
    """Calcula las metricas de calidad sobre la secuencia [T, H, W] gris."""
    # --- brillo / negro ---
    luma_por_frame = frames.mean(axis=(1, 2))
    luma_media = float(luma_por_frame.mean())
    frac_oscuros = float((luma_por_frame < UMBRAL_LUMA_FRAME_NEGRA).mean())

    # --- movimiento global (diff temporal media) ---
    if len(frames) >= 2:
        movimiento_global = float(np.abs(np.diff(frames, axis=0)).mean())
    else:
        movimiento_global = 0.0

    # --- region de la boca (centro del ROI) ---
    f0, f1, c0, c1 = CAJA_BOCA
    boca = frames[:, f0:f1, c0:c1]
    # actividad: cuanto varia cada pixel de la boca en el tiempo
    actividad_boca = float(boca.std(axis=0).mean()) if len(boca) >= 2 else 0.0
    # textura: nitidez/bordes (lengua, dientes, labios marcan bordes; piel/mano no)
    laps = [cv2.Laplacian(f, cv2.CV_32F).var() for f in boca]
    textura_boca = float(np.mean(laps)) if laps else 0.0

    return {
        "luma_media": round(luma_media, 2),
        "frac_oscuros": round(frac_oscuros, 3),
        "movimiento_global": round(movimiento_global, 3),
        "actividad_boca": round(actividad_boca, 3),
        "textura_boca": round(textura_boca, 2),
    }


def clasificar(m, texto):
    """Devuelve (estado, razones) a partir de las metricas y el texto del clip."""
    razones = []

    # DROP: claramente inservible
    es_negro = m["luma_media"] < UMBRAL_LUMA_CLIP or m["frac_oscuros"] > UMBRAL_FRAC_OSCUROS
    es_congelado = m["movimiento_global"] < UMBRAL_MOV_GLOBAL
    if es_negro:
        razones.append("negro")
    if es_congelado:
        razones.append("congelado")

    # REVIEW: senal mas blanda (boca tapada / mala alineacion / silencio)
    n_palabras = len(texto.split())
    boca_inactiva = (
        n_palabras >= MIN_PALABRAS_TEXTO
        and m["actividad_boca"] < UMBRAL_ACTIVIDAD_BOCA
        and m["textura_boca"] < UMBRAL_TEXTURA_BOCA
    )
    if boca_inactiva:
        razones.append("boca_inactiva")

    if es_negro or es_congelado:
        estado = "drop"
    elif boca_inactiva:
        estado = "review"
    else:
        estado = "keep"
    return estado, razones


def leer_texto(clip_path):
    txt = os.path.splitext(clip_path)[0] + ".txt"
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def auditar_clip(titulo, base, clip_path):
    """Audita un clip y devuelve una fila completa para el manifest (siempre 11 columnas)."""
    texto = leer_texto(clip_path)
    frames = leer_frames_gris(clip_path)
    if frames is None:
        return [titulo, base, 0, "", "", "", "", "", "drop", "video_no_legible", texto]
    m = metricas_clip(frames)
    estado, razones = clasificar(m, texto)
    return [
        titulo, base, len(frames),
        m["luma_media"], m["frac_oscuros"], m["movimiento_global"],
        m["actividad_boca"], m["textura_boca"],
        estado, ";".join(razones), texto,
    ]


def auditar_fuente(titulo, ya_auditados):
    """Audita todos los clips de una fuente. Devuelve lista de filas para el manifest."""
    origen = os.path.join(ROIS_DIR, titulo)
    filas = []
    clips = sorted(f for f in os.listdir(origen) if f.endswith(".mp4"))
    for nombre in clips:
        base = os.path.splitext(nombre)[0]
        if (titulo, base) in ya_auditados:
            continue
        fila = auditar_clip(titulo, base, os.path.join(origen, nombre))
        filas.append(fila)
        if fila[8] != "keep":
            print(f"  [{fila[8]}] {nombre} -> {fila[9]}")
    return filas


def cargar_manifest_existente():
    filas, ya = [], set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            lector = csv.reader(f)
            next(lector, None)
            for fila in lector:
                if len(fila) >= 2:
                    filas.append(fila)
                    ya.add((fila[0], fila[1]))
    return filas, ya


def guardar_manifest(filas):
    filas.sort(key=lambda x: (x[0], x[1]))
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ENCABEZADO)
        w.writerows(filas)


def resumen(filas):
    from collections import Counter
    estados = Counter(f[8] for f in filas)
    razones = Counter()
    for f in filas:
        for r in (f[9] or "").split(";"):
            if r:
                razones[r] += 1
    print("\n=== resumen ===")
    print("total:", len(filas), dict(estados))
    if razones:
        print("razones:", dict(razones))


def materializar_dataset(filas):
    """Copia los clips con estado `keep` a dataset/ y escribe dataset/manifest.csv."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    copiados = 0
    filas_curadas = []
    for fila in filas:
        titulo, base, estado = fila[0], fila[1], fila[8]
        if estado != "keep":
            continue
        src_dir = os.path.join(ROIS_DIR, titulo)
        dst_dir = os.path.join(DATASET_DIR, titulo)
        os.makedirs(dst_dir, exist_ok=True)
        for ext in (".mp4", ".txt"):
            src = os.path.join(src_dir, base + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, base + ext))
        copiados += 1
        filas_curadas.append([titulo, base, fila[2], fila[10]])
    with open(DATASET_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["titulo", "clip", "n_frames", "texto"])
        w.writerows(sorted(filas_curadas, key=lambda x: (x[0], x[1])))
    print(f"\nDataset curado: {copiados} clips keep -> {DATASET_DIR}")


def main():
    args = [a for a in sys.argv[1:] if a != "--materializar"]
    solo_materializar = "--materializar" in sys.argv

    if not os.path.isdir(ROIS_DIR):
        print(f"ERROR: no existe '{ROIS_DIR}'. Corré primero el preprocesamiento visual.")
        sys.exit(1)

    filas, ya = cargar_manifest_existente()

    if not solo_materializar or not filas:
        titulos = [args[0]] if args else sorted(
            d for d in os.listdir(ROIS_DIR) if os.path.isdir(os.path.join(ROIS_DIR, d)))
        for titulo in titulos:
            print(f"\n=== {titulo} ===")
            filas += auditar_fuente(titulo, ya)
        guardar_manifest(filas)
        print(f"\nManifest de auditoria: {MANIFEST_PATH}")

    resumen(filas)

    if solo_materializar:
        materializar_dataset(filas)
    else:
        print("\n(Para armar el dataset curado: agregá --materializar)")


if __name__ == "__main__":
    main()
