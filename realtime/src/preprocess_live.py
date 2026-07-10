"""
Preproc visual EN VIVO para el servicio de VSR.

Toma un clip grabado del navegador (o una lista de cuadros RGB) y produce el ROI
labial `(T, 96, 96)` uint8 a 25 fps que espera el modelo — exactamente el mismo
formato que el dataset offline.

REUSA `visual_preprocessing` (no duplica): mismo FaceLandmarker, los mismos 4 puntos
estables y el mismo warp a cara media (`VideoProcess`). La única diferencia con
`preprocesar.py` es la entrada (cuadros en memoria en vez de archivos del dataset) y
que acá no escribimos manifests ni .npz a disco: devolvemos el array para inferir.

Nota sobre reproducibilidad: el dataset se regeneró con numpy<2 para landmarks
bit-idénticos. Acá corremos con numpy 2 (lo pide espnet); los landmarks pueden diferir
en el último decimal, pero la *distribución* del ROI (warp + crop + escala) es la misma,
que es lo que el modelo necesita. No regeneramos dataset desde acá.
"""

import os

import cv2
import numpy as np

# Reuso directo de la etapa offline (lógica reutilizable en src/, ver AGENTS.md).
from visual_preprocessing.src.preprocesar import (
    FPS_SALIDA,
    TAMANO_SALIDA,
    UMBRAL_DETECCION,
    crear_landmarker,
    cuatro_puntos,
    detectar_landmarks,
    remuestrear_a_25fps,
)
from visual_preprocessing.src.video_process import VideoProcess

# Singletons por proceso: crear el landmarker es caro, se reusa entre requests.
_LANDMARKER = None
_VPROC = None


def _recursos():
    global _LANDMARKER, _VPROC
    if _LANDMARKER is None:
        _LANDMARKER = crear_landmarker()
        _VPROC = VideoProcess(crop_width=TAMANO_SALIDA, crop_height=TAMANO_SALIDA,
                              convert_gray=True)
    return _LANDMARKER, _VPROC


def rois_desde_cuadros(frames_rgb, fps_origen):
    """Lista de cuadros RGB (H,W,3) uint8 + fps -> (rois (T,96,96) uint8, ratio_deteccion).

    rois es None si no se detectó cara frontal estable en suficientes cuadros
    (mismo umbral que el dataset). El warp interpola cuadros sueltos sin detección.
    """
    landmarker, vproc = _recursos()

    puntos = []
    detectados = 0
    for rgb in frames_rgb:
        rgb = np.ascontiguousarray(rgb)
        landmarks = detectar_landmarks(rgb, landmarker)
        if landmarks is not None:
            puntos.append(cuatro_puntos(landmarks, rgb.shape[1], rgb.shape[0]))
            detectados += 1
        else:
            puntos.append(None)

    total = len(frames_rgb)
    ratio = detectados / total if total else 0.0
    if total == 0 or ratio < UMBRAL_DETECCION:
        return None, ratio

    try:
        secuencia = vproc(frames_rgb, puntos)
    except Exception:
        return None, ratio
    if secuencia is None or len(secuencia) == 0:
        return None, ratio

    frames = [secuencia[i] for i in range(len(secuencia))]
    rois = np.asarray(remuestrear_a_25fps(frames, fps_origen), dtype=np.uint8)  # (T,96,96)
    return rois, ratio


def leer_cuadros(video_path):
    """Lee un video (webm/mp4 del navegador) -> (cuadros RGB, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_SALIDA
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def rois_desde_video(video_path):
    """Clip grabado -> (rois (T,96,96) uint8 | None, ratio, n_cuadros_leidos, fps)."""
    frames, fps = leer_cuadros(video_path)
    rois, ratio = rois_desde_cuadros(frames, fps)
    return rois, ratio, len(frames), fps
