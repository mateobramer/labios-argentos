"""
Vista previa de un clip procesado.

Arma una hoja de contactos con cuadros del recorte labial para revisar a ojo.

Uso desde la raiz del repo:
    python -m visual_preprocessing.src.vista_previa "data/processed/lip_rois/<titulo>/clip_0010.mp4"
    python -m visual_preprocessing.src.vista_previa "data/processed/lip_rois/<titulo>/clip_0010.mp4" salida.png
"""

import os
import sys

import cv2
import numpy as np


COLS = 8
MAX_FILAS = 4
ESCALA = 3

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIEWS_DIR = os.path.join(DIR_MODULO, "outputs", "previews")


def leer_cuadros_gris(clip_path):
    cap = cv2.VideoCapture(clip_path)
    cuadros = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cuadros.append(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return cuadros


def ruta_preview_default(clip_path):
    titulo = os.path.basename(os.path.dirname(clip_path))
    nombre = os.path.splitext(os.path.basename(clip_path))[0] + "_preview.png"
    salida_dir = os.path.join(PREVIEWS_DIR, titulo)
    os.makedirs(salida_dir, exist_ok=True)
    return os.path.join(salida_dir, nombre)


def hoja_contactos(clip_path, salida=None):
    cuadros = leer_cuadros_gris(clip_path)
    if not cuadros:
        print(f"No se pudieron leer cuadros de: {clip_path}")
        return None

    n = min(len(cuadros), COLS * MAX_FILAS)
    idxs = np.linspace(0, len(cuadros) - 1, n).astype(int)
    seleccion = [cuadros[i] for i in idxs]

    h, w = seleccion[0].shape
    filas = (len(seleccion) + COLS - 1) // COLS
    while len(seleccion) < filas * COLS:
        seleccion.append(np.zeros((h, w), np.uint8))

    grilla = np.vstack([
        np.hstack(seleccion[r * COLS:(r + 1) * COLS])
        for r in range(filas)
    ])
    grilla = cv2.resize(
        grilla,
        None,
        fx=ESCALA,
        fy=ESCALA,
        interpolation=cv2.INTER_NEAREST,
    )

    if salida is None:
        salida = ruta_preview_default(clip_path)
    cv2.imwrite(salida, grilla)
    print(f"Vista previa guardada: {salida} ({len(cuadros)} cuadros)")
    return salida


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m visual_preprocessing.src.vista_previa <clip.mp4> [salida.png]")
        sys.exit(1)
    clip = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) >= 3 else None
    hoja_contactos(clip, salida)


if __name__ == "__main__":
    main()
