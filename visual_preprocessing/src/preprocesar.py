"""
Preprocesamiento visual de clips para VSR.

Entrada:
    data/clips/<titulo>/clip_NNNN.mp4

Salida:
    data/processed/lip_rois/<titulo>/clip_NNNN.mp4
    data/processed/lip_rois/<titulo>/clip_NNNN.txt
    data/metadata/lip_preprocessing_manifest.csv

Uso desde la raiz del repo:
    python -m visual_preprocessing.src.preprocesar
    python -m visual_preprocessing.src.preprocesar "<titulo>"
"""

import csv
import os
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("ERROR: falta 'mediapipe'. Instalar con: pip install -r visual_preprocessing/requirements.txt")
    sys.exit(1)


# Parametros del formato de salida. Confirmar contra el loader de entrenamiento.
TAMANO_SALIDA = 96
FPS_SALIDA = 25
MARGEN = 0.6
UMBRAL_DETECCION = 0.8

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ_REPO = os.path.dirname(DIR_MODULO)

CLIPS_DIR = os.path.join(RAIZ_REPO, "data", "clips")
SALIDA_DIR = os.path.join(RAIZ_REPO, "data", "processed", "lip_rois")
MANIFEST_PATH = os.path.join(RAIZ_REPO, "data", "metadata", "lip_preprocessing_manifest.csv")
MODELO = os.path.join(DIR_MODULO, "models", "face_landmarker.task")

# Indices de landmarks de labios en la malla facial de MediaPipe.
LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 80, 191, 78,
]


def crear_landmarker():
    """Crea el FaceLandmarker en modo imagen, cuadro a cuadro."""
    if not os.path.exists(MODELO):
        print(f"ERROR: falta el modelo '{MODELO}'.")
        print("Bajarlo con:")
        print("  curl -L -o visual_preprocessing/models/face_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
        sys.exit(1)
    base = mp_tasks.BaseOptions(model_asset_path=MODELO)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def caja_boca(landmarks, ancho, alto):
    """Devuelve (cx, cy, lado) de un recorte cuadrado centrado en la boca."""
    xs = [landmarks[i].x * ancho for i in LIPS_IDX]
    ys = [landmarks[i].y * alto for i in LIPS_IDX]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado = max(x1 - x0, y1 - y0) * (1 + MARGEN)
    return cx, cy, lado


def recortar(frame, caja):
    """Recorta la boca y la lleva a 96x96 en escala de grises."""
    alto, ancho = frame.shape[:2]
    cx, cy, lado = caja
    medio = lado / 2
    x0 = int(max(0, cx - medio))
    y0 = int(max(0, cy - medio))
    x1 = int(min(ancho, cx + medio))
    y1 = int(min(alto, cy + medio))
    recorte = frame[y0:y1, x0:x1]
    if recorte.size == 0:
        return None
    recorte = cv2.resize(recorte, (TAMANO_SALIDA, TAMANO_SALIDA))
    return cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)


def remuestrear_a_25fps(frames, fps_origen):
    """Selecciona cuadros para llevar la secuencia a FPS_SALIDA."""
    if not frames:
        return []
    if fps_origen <= 0:
        fps_origen = FPS_SALIDA
    duracion = len(frames) / fps_origen
    n_salida = max(1, round(duracion * FPS_SALIDA))
    salida = []
    for k in range(n_salida):
        idx = min(len(frames) - 1, round(k * fps_origen / FPS_SALIDA))
        salida.append(frames[idx])
    return salida


def detectar_labios(frame, landmarker):
    """Devuelve los landmarks de la primera cara, o None si no detecta."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(img)
    if res.face_landmarks:
        return res.face_landmarks[0]
    return None


def procesar_clip(clip_path, landmarker):
    """Lee un clip y devuelve (frames_recortados, ratio_deteccion)."""
    cap = cv2.VideoCapture(clip_path)
    fps_origen = cap.get(cv2.CAP_PROP_FPS)

    recortados = []
    detectados = 0
    total = 0
    ultima_caja = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        landmarks = detectar_labios(frame, landmarker)
        if landmarks is not None:
            ultima_caja = caja_boca(landmarks, frame.shape[1], frame.shape[0])
            detectados += 1

        if ultima_caja is None:
            continue
        gris = recortar(frame, ultima_caja)
        if gris is not None:
            recortados.append(gris)

    cap.release()
    ratio = detectados / total if total else 0.0
    return remuestrear_a_25fps(recortados, fps_origen), ratio


def guardar_video_gris(frames, salida_path):
    """Escribe una lista de cuadros 96x96 gris como mp4 a 25 fps."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(
        salida_path,
        fourcc,
        FPS_SALIDA,
        (TAMANO_SALIDA, TAMANO_SALIDA),
        isColor=False,
    )
    for frame in frames:
        vw.write(frame)
    vw.release()


def cargar_manifest():
    filas = []
    ya_procesados = set()
    if not os.path.exists(MANIFEST_PATH):
        return filas, ya_procesados

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            filas.append(fila)
            ya_procesados.add((fila["titulo"], fila["clip"]))
    return filas, ya_procesados


def procesar_carpeta(titulo, landmarker, ya_procesados, filas):
    origen = os.path.join(CLIPS_DIR, titulo)
    destino = os.path.join(SALIDA_DIR, titulo)
    os.makedirs(destino, exist_ok=True)

    clips = sorted(f for f in os.listdir(origen) if f.endswith(".mp4"))
    for nombre in clips:
        base = os.path.splitext(nombre)[0]
        if (titulo, base) in ya_procesados:
            print(f"  [salteado] {nombre} (ya procesado)")
            continue

        clip_path = os.path.join(origen, nombre)
        txt_origen = os.path.join(origen, base + ".txt")
        texto = ""
        if os.path.exists(txt_origen):
            with open(txt_origen, encoding="utf-8") as f:
                texto = f.read().strip()

        frames, ratio = procesar_clip(clip_path, landmarker)
        salida_mp4 = os.path.join(destino, base + ".mp4")
        salida_txt = os.path.join(destino, base + ".txt")

        if ratio < UMBRAL_DETECCION or not frames:
            estado = "descartado"
            print(f"  [descartado] {nombre} (cara en {ratio*100:.0f}% de los cuadros)")
        else:
            estado = "ok"
            guardar_video_gris(frames, salida_mp4)
            with open(salida_txt, "w", encoding="utf-8") as f:
                f.write(texto)
            print(f"  [ok] {nombre} -> {len(frames)} cuadros 96x96 gris")

        filas.append({
            "titulo": titulo,
            "clip": base,
            "clip_path": os.path.join("data", "clips", titulo, nombre),
            "text_path": os.path.join("data", "clips", titulo, base + ".txt"),
            "output_path": os.path.join("data", "processed", "lip_rois", titulo, base + ".mp4"),
            "estado": estado,
            "ratio_deteccion": f"{ratio:.3f}",
            "n_frames": str(len(frames)),
            "texto": texto,
        })


def guardar_manifest(filas):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    encabezado = [
        "titulo", "clip", "clip_path", "text_path", "output_path",
        "estado", "ratio_deteccion", "n_frames", "texto",
    ]
    filas.sort(key=lambda x: (x["titulo"], x["clip"]))
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f_csv:
        escritor = csv.DictWriter(f_csv, fieldnames=encabezado)
        escritor.writeheader()
        escritor.writerows(filas)


def main():
    if not os.path.isdir(CLIPS_DIR):
        print(f"ERROR: no existe la carpeta '{CLIPS_DIR}'. Corre primero descargar_procesar.py.")
        sys.exit(1)

    if len(sys.argv) >= 2:
        titulos = [sys.argv[1]]
    else:
        titulos = sorted(
            d for d in os.listdir(CLIPS_DIR)
            if os.path.isdir(os.path.join(CLIPS_DIR, d))
        )

    os.makedirs(SALIDA_DIR, exist_ok=True)
    filas, ya_procesados = cargar_manifest()
    landmarker = crear_landmarker()
    try:
        for titulo in titulos:
            print(f"\n=== {titulo} ===")
            procesar_carpeta(titulo, landmarker, ya_procesados, filas)
    finally:
        landmarker.close()

    guardar_manifest(filas)
    print(f"\nListo. Manifest en: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
