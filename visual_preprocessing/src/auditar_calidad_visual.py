"""
Metricas visuales con MediaPipe para decidir calidad VSR.

No borra clips. Toma muestras de frames y mide:
- cuantas caras aparecen;
- si hay frames sin cara aunque haya texto;
- si aparecen multiples caras;
- si la boca de la cara principal parece visible;
- si la boca esta razonablemente centrada.
"""

from pathlib import Path

import cv2
import numpy as np


RAIZ_REPO = Path(__file__).resolve().parents[2]
MODELO = RAIZ_REPO / "visual_preprocessing" / "models" / "face_landmarker.task"

LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 80, 191, 78,
]


def importar_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        raise RuntimeError(
            "Falta mediapipe. Instalar en un entorno para preprocesamiento con: "
            "pip install -r visual_preprocessing/requirements.txt"
        ) from exc
    return mp, mp_tasks, mp_vision


def crear_landmarker(num_faces=3):
    if not MODELO.exists():
        raise FileNotFoundError(f"No existe el modelo MediaPipe: {MODELO}")

    mp, mp_tasks, mp_vision = importar_mediapipe()
    base = mp_tasks.BaseOptions(model_asset_path=str(MODELO))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=num_faces,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts), mp


def indices_muestra(frame_count, n_frames):
    if frame_count <= 0:
        return []
    n = min(n_frames, frame_count)
    return sorted(set(np.linspace(0, frame_count - 1, n).astype(int).tolist()))


def leer_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def detectar_caras(frame, landmarker, mp):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(img)
    return res.face_landmarks or []


def metricas_boca(landmarks, ancho, alto):
    xs = np.array([landmarks[i].x * ancho for i in LIPS_IDX])
    ys = np.array([landmarks[i].y * alto for i in LIPS_IDX])
    dentro = ((xs >= 0) & (xs < ancho) & (ys >= 0) & (ys < alto)).mean()

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    cx = (x0 + x1) / 2 / ancho
    cy = (y0 + y1) / 2 / alto

    boca_visible = dentro >= 0.8 and w / ancho >= 0.025 and h / alto >= 0.01
    distancia_centro = abs(cx - 0.5) + abs(cy - 0.58)
    centrada = distancia_centro <= 0.45

    return {
        "boca_visible": boca_visible,
        "boca_centrada": centrada,
        "boca_x": cx,
        "boca_y": cy,
        "boca_w": w / ancho,
        "boca_h": h / alto,
    }


def analizar_clip(clip_path, texto="", n_frames=7, landmarker=None, mp=None):
    clip_path = Path(clip_path)
    cerrar_landmarker = False
    if landmarker is None or mp is None:
        landmarker, mp = crear_landmarker()
        cerrar_landmarker = True

    cap = cv2.VideoCapture(str(clip_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    indices = indices_muestra(frame_count, n_frames)

    frames_leidos = 0
    conteos_caras = []
    bocas_visibles = []
    bocas_centradas = []
    centros_x = []
    centros_y = []

    for idx in indices:
        frame = leer_frame(cap, idx)
        if frame is None:
            continue
        frames_leidos += 1
        alto, ancho = frame.shape[:2]
        caras = detectar_caras(frame, landmarker, mp)
        conteos_caras.append(len(caras))
        if len(caras) == 1:
            m = metricas_boca(caras[0], ancho, alto)
            bocas_visibles.append(m["boca_visible"])
            bocas_centradas.append(m["boca_centrada"])
            centros_x.append(m["boca_x"])
            centros_y.append(m["boca_y"])

    cap.release()
    if cerrar_landmarker:
        landmarker.close()

    frames_con_cara = sum(c > 0 for c in conteos_caras)
    frames_multi_cara = sum(c > 1 for c in conteos_caras)
    max_caras = max(conteos_caras) if conteos_caras else 0
    ratio_cara = frames_con_cara / frames_leidos if frames_leidos else 0.0
    ratio_multi = frames_multi_cara / frames_leidos if frames_leidos else 0.0
    ratio_boca_visible = sum(bocas_visibles) / len(bocas_visibles) if bocas_visibles else 0.0
    ratio_boca_centrada = sum(bocas_centradas) / len(bocas_centradas) if bocas_centradas else 0.0
    duracion = frame_count / fps if fps > 0 else 0.0

    razones = []
    if frames_leidos == 0:
        razones.append("video_no_legible")
    if texto.strip() and ratio_cara == 0:
        razones.append("sin_rostro_con_texto")
    elif ratio_cara < 0.6:
        razones.append("rostro_inestable")
    if ratio_multi > 0:
        razones.append("multiples_rostros")
    if ratio_cara > 0 and ratio_boca_visible < 0.6:
        razones.append("boca_poco_visible")
    if ratio_boca_visible > 0 and ratio_boca_centrada < 0.6:
        razones.append("boca_descentrada")

    return {
        "clip_path": str(clip_path),
        "frames_muestreados": frames_leidos,
        "duration_sec": round(duracion, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "max_caras": max_caras,
        "ratio_cara": round(ratio_cara, 3),
        "ratio_multi_cara": round(ratio_multi, 3),
        "ratio_boca_visible": round(ratio_boca_visible, 3),
        "ratio_boca_centrada": round(ratio_boca_centrada, 3),
        "boca_x_promedio": round(float(np.mean(centros_x)), 3) if centros_x else "",
        "boca_y_promedio": round(float(np.mean(centros_y)), 3) if centros_y else "",
        "visual_status": "review" if razones else "keep_candidate",
        "visual_reasons": ";".join(razones),
    }
