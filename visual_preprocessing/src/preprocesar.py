"""
Preprocesamiento visual de clips para VSR (formato Auto-AVSR).

Entrada:
    data/clips/<titulo>/clip_NNNN.mp4

Salida:
    data/processed/lip_rois/<titulo>/clip_NNNN.mp4   (recorte labial 96x96 gris, 25 fps; para QA visual)
    data/processed/lip_rois/<titulo>/clip_NNNN.npz   (mismo recorte como array (T,96,96) uint8, sin perdida; entrada del modelo)
    data/processed/lip_rois/<titulo>/clip_NNNN.txt
    data/metadata/lip_preprocessing_manifest.csv

Uso desde la raiz del repo:
    python -m visual_preprocessing.src.preprocesar                      # todas las fuentes
    python -m visual_preprocessing.src.preprocesar "<titulo>" ["<titulo2>" ...]
    python -m visual_preprocessing.src.preprocesar --jobs 7             # N procesos en paralelo

Reanudable e idempotente: saltea los clips que ya tienen .npz, asi que si se corta
(apagon, Ctrl-C) se vuelve a correr el mismo comando y retoma donde quedo.

Como funciona (alineacion a cara media, estilo Auto-AVSR):
    1. MediaPipe FaceLandmarker detecta 478 landmarks por cuadro.
    2. De esos puntos se extraen 4 puntos estables: ojo derecho, ojo izquierdo,
       punta de nariz y centro de boca.
    3. VideoProcess (video_process.py, adaptado de Auto-AVSR) usa esos 4 puntos para
       una transformacion afin que alinea cada cuadro a una cara canonica (pose, escala
       y rotacion normalizadas), pasa a gris y recorta 96x96 centrado en la boca.
    4. Se descartan los clips donde no se detecta cara frontal estable.

Diferencia con el recorte por bounding-box: el warp a cara media deja la boca siempre
en la misma posicion/escala/orientacion, que es lo que el modelo de Auto-AVSR espera.
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("ERROR: falta 'mediapipe'. Instalar con: pip install -r visual_preprocessing/requirements.txt")
    sys.exit(1)

from visual_preprocessing.src.video_process import VideoProcess

# Parametros del formato de salida. Confirmar contra el loader de entrenamiento.
TAMANO_SALIDA = 96
FPS_SALIDA = 25
UMBRAL_DETECCION = 0.8  # fraccion minima de cuadros con cara para conservar el clip

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ_REPO = os.path.dirname(DIR_MODULO)

CLIPS_DIR = os.path.join(RAIZ_REPO, "data", "clips")
SALIDA_DIR = os.path.join(RAIZ_REPO, "data", "processed", "lip_rois")
MANIFEST_PATH = os.path.join(RAIZ_REPO, "data", "metadata", "lip_preprocessing_manifest.csv")
MODELO = os.path.join(DIR_MODULO, "models", "face_landmarker.task")

# Indices de la malla facial de MediaPipe (478 puntos) para los 4 puntos estables.
# El orden DEBE ser [ojo derecho, ojo izquierdo, nariz, boca] para que matchee la
# referencia de cara media usada en VideoProcess.
OJO_DER_IDX = [33, 133, 159, 145]    # ojo derecho del sujeto (lado izquierdo de la imagen)
OJO_IZQ_IDX = [362, 263, 386, 374]   # ojo izquierdo del sujeto
NARIZ_IDX = 1                        # punta de nariz
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


def detectar_landmarks(rgb, landmarker):
    """Devuelve los 478 landmarks de la primera cara (en RGB), o None."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(img)
    if res.face_landmarks:
        return res.face_landmarks[0]
    return None


def cuatro_puntos(landmarks, ancho, alto):
    """Extrae [ojo derecho, ojo izquierdo, nariz, boca] en pixeles desde la malla."""
    def media(idxs):
        return [float(np.mean([landmarks[i].x for i in idxs])) * ancho,
                float(np.mean([landmarks[i].y for i in idxs])) * alto]

    return np.array([
        media(OJO_DER_IDX),
        media(OJO_IZQ_IDX),
        [landmarks[NARIZ_IDX].x * ancho, landmarks[NARIZ_IDX].y * alto],
        media(LIPS_IDX),
    ], dtype=np.float32)


def remuestrear_a_25fps(frames, fps_origen):
    """Selecciona cuadros para llevar la secuencia a FPS_SALIDA por nearest-frame."""
    if len(frames) == 0:
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


def procesar_clip(clip_path, landmarker, vproc):
    """Lee un clip, detecta 4 puntos por cuadro y alinea+recorta con warp a cara media.

    Devuelve (frames_recortados, ratio_deteccion): lista de imagenes 96x96 gris
    (alineadas y remuestreadas a 25 fps) y la fraccion de cuadros con cara detectada.
    """
    cap = cv2.VideoCapture(clip_path)
    fps_origen = cap.get(cv2.CAP_PROP_FPS)

    frames_rgb = []
    puntos = []
    detectados = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames_rgb.append(rgb)
        landmarks = detectar_landmarks(rgb, landmarker)
        if landmarks is not None:
            puntos.append(cuatro_puntos(landmarks, rgb.shape[1], rgb.shape[0]))
            detectados += 1
        else:
            puntos.append(None)

    cap.release()
    total = len(frames_rgb)
    ratio = detectados / total if total else 0.0

    if total == 0 or ratio < UMBRAL_DETECCION:
        return [], ratio

    # Warp a cara media + recorte 96x96 gris (interpola cuadros sin deteccion).
    try:
        secuencia = vproc(frames_rgb, puntos)
    except Exception as e:
        print(f"    (warp fallo: {e})")
        return [], ratio

    if secuencia is None or len(secuencia) == 0:
        return [], ratio

    frames = [secuencia[i] for i in range(len(secuencia))]
    return remuestrear_a_25fps(frames, fps_origen), ratio


def guardar_video_gris(frames, salida_path):
    """Escribe una lista de cuadros 96x96 gris como mp4 a 25 fps (solo para QA visual)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(salida_path, fourcc, FPS_SALIDA,
                         (TAMANO_SALIDA, TAMANO_SALIDA), isColor=False)
    for f in frames:
        vw.write(f)
    vw.release()


def guardar_npz(frames, salida_path):
    """Guarda la secuencia como array (T,96,96) uint8 sin perdida.

    Es la entrada real del modelo (el mp4 es solo para inspeccion visual): el codec
    mp4v comprime con perdida y mete artefactos de bloque en el recorte chico, lo que
    contaminaria el WER. El .npz conserva el recorte exacto que produjo el warp.
    """
    arr = np.asarray(frames, dtype=np.uint8)  # (T, 96, 96)
    np.savez_compressed(salida_path, rois=arr)


# --- Procesamiento paralelo, un landmarker persistente por worker ---
#
# La verdad de "que esta hecho" es el .npz en disco: el skip y la reanudacion se
# basan en su existencia (no en el manifest), asi que es parallel-safe y sobrevive
# a un apagon. El manifest es solo un log; se checkpointea cada 50 clips.

_W_LANDMARKER = None
_W_VPROC = None


def _init_worker():
    """Inicializa el landmarker y el warp una sola vez por proceso worker."""
    global _W_LANDMARKER, _W_VPROC
    _W_LANDMARKER = crear_landmarker()
    _W_VPROC = VideoProcess(crop_width=TAMANO_SALIDA, crop_height=TAMANO_SALIDA,
                            convert_gray=True)


def _procesar_clip_tarea(tarea):
    """Worker: procesa un clip, escribe sus salidas y devuelve la fila del manifest."""
    titulo, base = tarea
    origen = os.path.join(CLIPS_DIR, titulo)
    destino = os.path.join(SALIDA_DIR, titulo)
    clip_path = os.path.join(origen, base + ".mp4")

    frames, ratio = procesar_clip(clip_path, _W_LANDMARKER, _W_VPROC)

    txt_origen = os.path.join(origen, base + ".txt")
    texto = ""
    if os.path.exists(txt_origen):
        with open(txt_origen, encoding="utf-8") as f:
            texto = f.read().strip()

    if not frames:
        estado = "descartado"
    else:
        estado = "ok"
        os.makedirs(destino, exist_ok=True)
        guardar_video_gris(frames, os.path.join(destino, base + ".mp4"))
        guardar_npz(frames, os.path.join(destino, base + ".npz"))
        with open(os.path.join(destino, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(texto)
    return [titulo, base, estado, f"{ratio:.3f}", str(len(frames)), texto]


def npz_existe(titulo, base):
    return os.path.exists(os.path.join(SALIDA_DIR, titulo, base + ".npz"))


def cargar_manifest():
    """Lee el manifest existente como dict {(titulo, base): fila}. Vacio si no hay."""
    filas = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            lector = csv.reader(f)
            next(lector, None)
            for fila in lector:
                if len(fila) >= 2:
                    filas[(fila[0], fila[1])] = fila
    return filas


def escribir_manifest(filas_dict):
    """Escribe el manifest de forma atomica (tmp + replace) para que nunca quede a medias."""
    encabezado = ["titulo", "clip", "estado", "ratio_deteccion", "n_frames", "texto"]
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(encabezado)
        escritor.writerows(sorted(filas_dict.values(), key=lambda x: (x[0], x[1])))
    os.replace(tmp, MANIFEST_PATH)


def listar_pendientes(titulos, manifest):
    """Clips a procesar: los que no tienen .npz y no figuran ya como descartados."""
    limite = int(os.environ.get("PREPROC_MAX", "0"))
    descartados = {k for k, v in manifest.items() if len(v) > 2 and v[2] == "descartado"}
    pendientes = []
    for titulo in titulos:
        origen = os.path.join(CLIPS_DIR, titulo)
        if not os.path.isdir(origen):
            print(f"  (aviso) no existe la fuente: {titulo}")
            continue
        clips = sorted(f for f in os.listdir(origen) if f.endswith(".mp4"))
        if limite > 0:
            clips = clips[:limite]
        for nombre in clips:
            base = os.path.splitext(nombre)[0]
            if npz_existe(titulo, base) or (titulo, base) in descartados:
                continue
            pendientes.append((titulo, base))
    return pendientes


def reconciliar_manifest(titulos, manifest):
    """Deja el manifest igual al disco para las fuentes tocadas: agrega filas para
    los .npz sin entrada y descarta filas 'ok' cuyo .npz ya no existe (limpia
    nombres viejos/stale). No toca fuentes que no esten en `titulos`."""
    objetivo = set(titulos)
    # Descartar filas 'ok' sin .npz de las fuentes que estamos procesando.
    for clave in [k for k, v in manifest.items()
                  if k[0] in objetivo and len(v) > 2 and v[2] == "ok"
                  and not npz_existe(*k)]:
        del manifest[clave]
    # Agregar filas faltantes para cada .npz en disco.
    for titulo in titulos:
        destino = os.path.join(SALIDA_DIR, titulo)
        if not os.path.isdir(destino):
            continue
        for npz in sorted(f for f in os.listdir(destino) if f.endswith(".npz")):
            base = os.path.splitext(npz)[0]
            if (titulo, base) in manifest:
                continue
            try:
                n = int(np.load(os.path.join(destino, npz))["rois"].shape[0])
            except Exception:
                n = 0
            txt = os.path.join(destino, base + ".txt")
            texto = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
            manifest[(titulo, base)] = [titulo, base, "ok", "", str(n), texto]


def procesar_paralelo(titulos, jobs):
    manifest = cargar_manifest()
    pendientes = listar_pendientes(titulos, manifest)
    total = len(pendientes)
    if total == 0:
        print("No hay clips pendientes (todos tienen .npz). Nada que hacer.")
        reconciliar_manifest(titulos, manifest)
        escribir_manifest(manifest)
        return

    print(f"Procesando {total} clips pendientes con {jobs} worker(s)...")
    hechos = ok = 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        futuros = [ex.submit(_procesar_clip_tarea, t) for t in pendientes]
        for fut in as_completed(futuros):
            fila = fut.result()
            manifest[(fila[0], fila[1])] = fila
            hechos += 1
            ok += (fila[2] == "ok")
            if hechos % 50 == 0:
                escribir_manifest(manifest)  # checkpoint del log
                print(f"  {hechos}/{total}  ({ok} ok, {hechos - ok} descartados)")
    reconciliar_manifest(titulos, manifest)
    escribir_manifest(manifest)
    print(f"\nListo. {hechos} clips ({ok} ok, {hechos - ok} descartados).")
    print(f"Manifest: {MANIFEST_PATH}")


def main():
    if not os.path.isdir(CLIPS_DIR):
        print(f"ERROR: no existe '{CLIPS_DIR}'. Corré primero descargar_procesar.py.")
        sys.exit(1)

    args = sys.argv[1:]
    jobs = 1
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = max(1, int(args[i + 1]))
        del args[i:i + 2]

    if args:
        titulos = args  # una o varias fuentes explicitas
    else:
        titulos = sorted(d for d in os.listdir(CLIPS_DIR)
                         if os.path.isdir(os.path.join(CLIPS_DIR, d)))

    procesar_paralelo(titulos, jobs)


if __name__ == "__main__":
    main()
