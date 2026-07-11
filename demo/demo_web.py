#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UI web para la demo de VSR en vivo (VAD visual + transcript acumulado).

Mismo motor que demo_stream.py (landmarks en vivo -> VAD por pausas de labios ->
segmentos -> ViSpeR beam3 con encoder en MPS -> texto), pero en vez de una ventana
cv2 sirve una pagina linda en http://localhost:8551 :
  - stream de camara (MJPEG), medidor de labios con umbral, chip de estado
  - captions grandes + panel de transcript con duracion/latencia por segmento
  - botones copiar / limpiar. Renderiza ñ y acentos (cv2 no podia).

Solo stdlib (http.server) — sin dependencias nuevas. Corre en el env `ptt`:
  ~/miniconda3/envs/ptt/bin/python demo_web.py [--port 8551] [--max-seg 4.0]
                                               [--pause 0.45] [--sens 3.0] [--qwen]
Ctrl-C para salir.
"""
import os, re, sys, time, json, threading, subprocess, tempfile, argparse, collections, statistics, webbrowser, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

# Raiz del repo: por default se deriva de la ubicacion de este archivo (funciona
# desde cualquier clone); LABIOS_REPO la pisa si hace falta. Ver .env.example.
REPO = os.environ.get("LABIOS_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from preprocessing.src.preprocesar import (
    crear_landmarker, detectar_landmarks, cuatro_puntos, remuestrear_a_25fps)
from preprocessing.src.video_process import VideoProcess

VISPER_PY = os.path.expanduser(os.environ.get("VISPER_PY", "~/miniconda3/envs/visper/bin/python"))
INFER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_server.py")
HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
HTML_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "modelo.html")
MODELOS_PERSONAL = os.path.join(REPO, "modelos", "personal")
MIN_SEG_S, PREROLL_S, MOV_WIN_S, CALIB_S = 0.7, 0.25, 0.35, 2.0
RUN_ID = str(int(time.time()))   # cambia por arranque: bustea la cache de /strip en el navegador

# ---- estado compartido (los handlers http lo leen) ----
S = {"fase": "calibrando", "mov": 0.0, "thr": None, "busy": False,
     "segmentos": [], "config": {}, "ultimo_forzado": False,
     "n_caras": 0, "lock_bbox": None, "camara": "iniciando"}
JPEG = {"data": None}
STRIPS = []          # tiras "lo que ve el modelo": jpg de 7 bocas 96x96 por segmento
BUF = collections.deque()
LOCK = threading.Lock()
STOP = threading.Event()
SRV = {"proc": None, "lock": threading.Lock()}

# ---- modo calibracion (grabar frases para adaptar el modelo a una persona) ----
from personalization.build_testset import PROMPTS as CAL_PROMPTS  # set curado de 1100 frases versionado en personalization/ (REPO ya esta en sys.path)
from personalization.sesiones import Sesiones, nombre_persona, NIVELES
CAL_N = 40
PERSONAL_ROOT = os.environ.get("VSR_PERSONAL_DIR", "~/vsr_personal")
SESIONES = Sesiones(PERSONAL_ROOT, CAL_PROMPTS)
CAL = {"activo": False, "persona": "", "dir": None, "rec_t0": None,
       "pendiente": None, "preview": None, "msg": ""}
CAL_LOCK = threading.Lock()
TRAIN = {"proc": None, "fase": "idle", "salida": [], "error": ""}
TRAIN_LOCK = threading.Lock()

# ---- subida incremental: cada toma aceptada va al bucket en segundo plano, asi
# al tocar Entrenar ya esta casi todo arriba (rsync sube solo lo que falte) ----
CAL_BUCKET = os.environ.get("VSR_BUCKET", "gs://labios-argentos-vsr-clean-v1")
GCLOUD = shutil.which("gcloud")
SUBIDA = collections.deque()
SUBIDA_EV = threading.Event()

def hilo_subida_clips():
    while True:
        SUBIDA_EV.wait()
        while SUBIDA:
            ruta, persona = SUBIDA.popleft()
            try:
                subprocess.run([GCLOUD, "storage", "cp", ruta,
                                f"{CAL_BUCKET}/calibracion/{persona}/rois/"],
                               capture_output=True, timeout=600)
            except Exception as e:
                print(f"[web] subida en segundo plano fallo ({e}); rsync la cubre al entrenar", flush=True)
        SUBIDA_EV.clear()

def encolar_subida(carpeta, frase_id, persona):
    if not GCLOUD:
        return
    SUBIDA.append((os.path.join(str(carpeta), f"clip_{frase_id:03d}.npz"), persona))
    SUBIDA_EV.set()

def apertura(lm):
    boca = abs(lm[14].y - lm[13].y)
    cara = abs(lm[152].y - lm[10].y) + 1e-6
    return boca / cara

def norm_texto(s):
    """Misma normalizacion del dataset: minusculas, sin tildes (ñ preservada), solo [a-z0-9ñ ]."""
    import re as _re, unicodedata as _u
    s = s.lower().strip().replace("ñ", "\x00")
    s = _u.normalize("NFD", s)
    s = "".join(c for c in s if _u.category(c) != "Mn").replace("\x00", "ñ")
    return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9ñ ]", " ", s)).strip()

def lanzar_servidor(use_qwen):
    env = dict(os.environ)
    if use_qwen:
        env["VSR_QWEN"] = "1"
    # beam 5 siempre: permite prender/apagar qwen en runtime con las 5 candidatas completas
    # (beam5 vs beam3 = mismo WER, +0.15s; ver docs/experiments/09)
    env.setdefault("VSR_BEAM", "5")
    print("[web] cargando ViSpeR (primera vez ~20-40s, NO cortes)...", flush=True)
    p = subprocess.Popen([VISPER_PY, INFER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, env=env)
    cfg = {}
    for line in p.stdout:
        line = line.strip()
        if line.startswith("CONFIG "):
            try: cfg = json.loads(line[7:])
            except Exception: pass
        if line == "READY": break
    print(f"[web] server de inferencia listo: {cfg}", flush=True)
    return p, cfg

# --- seleccion de cara con VARIAS personas en camara ---
# Detectamos hasta 3 caras y elegimos con "sticky lock": al arrancar (o si se pierde >1.5s),
# lockeamos la cara MAS GRANDE (la mas cercana a camara); despues seguimos siempre a la mas
# cercana a la posicion lockeada, aunque entre otra cara mas grande al cuadro.
FACE_LOCK = {"c": None, "t": 0.0}

def elegir_cara(caras):
    if not caras:
        return None
    now = time.time()
    def geo(lm):  # (centro_x, centro_y, tamaño) en coords normalizadas
        return (lm[1].x, lm[1].y, abs(lm[152].y - lm[10].y))
    cands = [(geo(f), f) for f in caras]
    if FACE_LOCK["c"] is not None and (now - FACE_LOCK["t"]) < 1.5:
        (g, f) = min(cands, key=lambda cf: (cf[0][0]-FACE_LOCK["c"][0])**2 + (cf[0][1]-FACE_LOCK["c"][1])**2)
        d2 = (g[0]-FACE_LOCK["c"][0])**2 + (g[1]-FACE_LOCK["c"][1])**2
        if d2 < 0.04:                       # sigue siendo la misma persona
            FACE_LOCK["c"] = (g[0], g[1]); FACE_LOCK["t"] = now
            return f
    (g, f) = max(cands, key=lambda cf: cf[0][2])   # (re)lock: la mas grande
    FACE_LOCK["c"] = (g[0], g[1]); FACE_LOCK["t"] = now
    return f

def hilo_landmarks():
    from preprocessing.src import preprocesar as pp
    base = pp.mp_tasks.BaseOptions(model_asset_path=pp.MODELO)
    opts = pp.mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=3)
    lmk = pp.mp_vision.FaceLandmarker.create_from_options(opts)
    while not STOP.is_set():
        with LOCK:
            pend = [e for e in BUF if e["pts"] is False][:4]
        if not pend:
            time.sleep(0.01); continue
        for e in pend:
            rgb = np.ascontiguousarray(cv2.cvtColor(e["frame"], cv2.COLOR_BGR2RGB))
            img = pp.mp.Image(image_format=pp.mp.ImageFormat.SRGB, data=rgb)
            res = lmk.detect(img)
            caras = res.face_landmarks or []
            S["n_caras"] = len(caras)
            lm = elegir_cara(caras)
            if lm is not None:
                e["pts"] = cuatro_puntos(lm, rgb.shape[1], rgb.shape[0])
                e["open"] = apertura(lm)
                xs = [p.x for p in lm]; ys = [p.y for p in lm]
                S["lock_bbox"] = (min(xs), min(ys), max(xs), max(ys))
            else:
                e["pts"] = None; e["open"] = None
                S["lock_bbox"] = None

def hilo_segmentador(args):
    vproc = VideoProcess()
    npz_path = os.path.join(tempfile.gettempdir(), "web_seg.npz")
    aper = collections.deque()
    base, t_ini, thr = [], time.time(), None
    hablando, seg_t0, quieto_desde = False, 0.0, None
    consumido_hasta = 0.0

    def movimiento(t_now):
        vals = [o for (t, o) in aper if t >= t_now - MOV_WIN_S and o is not None]
        return statistics.pstdev(vals) if len(vals) >= 3 else 0.0

    def inferir(t_a, t_b, forzado):
        with LOCK:
            seg = [e for e in BUF if t_a <= e["t"] <= t_b and e["pts"] is not False]
        if not seg or (seg[-1]["t"] - seg[0]["t"]) < MIN_SEG_S: return
        dur = seg[-1]["t"] - seg[0]["t"]
        fps = (len(seg)-1) / max(dur, 1e-3)
        rgb = [np.ascontiguousarray(cv2.cvtColor(e["frame"], cv2.COLOR_BGR2RGB)) for e in seg]
        pts = [e["pts"] for e in seg]
        if sum(1 for p in pts if p is not None) / len(pts) < 0.5: return
        S["busy"] = True
        t0 = time.time()
        out = ""
        try:
            seq = vproc(rgb, pts)
            if seq is not None and len(seq) > 0:
                arr = np.asarray(remuestrear_a_25fps([seq[i] for i in range(len(seq))], fps), dtype=np.uint8)
                np.savez_compressed(npz_path, rois=arr)
                with SRV["lock"]:
                    SRV["proc"].stdin.write(npz_path + "\n"); SRV["proc"].stdin.flush()
                    out = SRV["proc"].stdout.readline().strip()
        except Exception as e:
            out = f"__ERROR__ {e}"
        dt = time.time() - t0
        if out and not out.startswith("__ERROR__"):
            palabras = out.split()
            if S["ultimo_forzado"] and S["segmentos"]:
                prev = S["segmentos"][-1]["texto"].split()
                if prev and palabras and prev[-1] == palabras[0]:
                    palabras = palabras[1:]
            if palabras:
                # tira de bocas (7 frames equiespaciados del ROI que VIO el modelo)
                strip_id = -1
                try:
                    idxs = np.linspace(0, len(arr) - 1, 7).astype(int)
                    tira = np.concatenate([arr[k] for k in idxs], axis=1)   # 96 x 672 gris
                    ok3, jt = cv2.imencode(".jpg", tira, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok3:
                        STRIPS.append(jt.tobytes()); strip_id = len(STRIPS) - 1
                except Exception:
                    pass
                S["segmentos"].append({"texto": " ".join(palabras),
                                       "dur": round(dur, 2), "infer": round(dt, 2),
                                       "strip": strip_id})
        S["ultimo_forzado"] = forzado
        S["busy"] = False

    while not STOP.is_set():
        if CAL["activo"]:                      # en modo calibracion: sin VAD ni inferencia
            time.sleep(0.1)
            with LOCK:                          # podar igual, pero respetando la grabacion activa
                lim = (CAL["rec_t0"] - 0.5) if CAL["rec_t0"] else (time.time() - 1.0)
                while BUF and BUF[0]["t"] < lim:
                    BUF.popleft()
            consumido_hasta = time.time()
            continue
        with LOCK:
            nuevos = [(e["t"], e["open"]) for e in BUF
                      if e["pts"] is not False and e["t"] > consumido_hasta]
        if not nuevos:
            time.sleep(0.015); continue
        for (t, o) in nuevos:
            consumido_hasta = t
            aper.append((t, o))
            while aper and aper[0][0] < t - max(MOV_WIN_S, 1.0):
                aper.popleft()
            if thr is None:                          # calibracion con silencio inicial
                if o is not None: base.append(o)
                if t - t_ini >= CALIB_S and len(base) >= 10:
                    thr = max(args.sens * statistics.pstdev(base[-40:]), 0.004)
                    S["thr"] = thr; S["fase"] = "escuchando"
                continue
            mov = movimiento(t); S["mov"] = round(mov, 5)
            if not hablando:
                if mov > thr:
                    hablando, seg_t0, quieto_desde = True, t - PREROLL_S, None
                    S["fase"] = "HABLANDO"
            else:
                if mov <= thr:
                    quieto_desde = quieto_desde or t
                    if t - quieto_desde >= args.pause:
                        hablando = False; S["fase"] = "escuchando"
                        inferir(seg_t0, quieto_desde + 0.1, forzado=False)
                else:
                    quieto_desde = None
                if hablando and t - seg_t0 >= args.max_seg:
                    inferir(seg_t0, t, forzado=True)
                    seg_t0 = t
        corte = (seg_t0 if hablando else consumido_hasta) - PREROLL_S - 0.5
        with LOCK:
            while BUF and BUF[0]["t"] < corte:
                BUF.popleft()

def hilo_camara():
    # En Windows DirectShow evita que Media Foundation abra una sesion vacia o
    # quede bloqueada despues de cerrar una instancia anterior de la demo.
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        S["camara"] = "No pude abrir la cámara. Cerrá otras apps que la estén usando y reiniciá la demo."
        print("[web] ERROR: no abre la camara"); return
    for _ in range(40):                              # warmup macOS
        ok, _f = cap.read()
        if ok: break
        time.sleep(0.2)
    if not ok:
        S["camara"] = "La cámara abrió pero no entrega imagen. Cerrá otras apps que la estén usando y reiniciá la demo."
        cap.release(); print("[web] ERROR: la camara no entrego cuadros"); return
    S["camara"] = "ok"
    print("[web] camara ok.", flush=True)
    last_jpg = 0.0
    fails = 0
    while not STOP.is_set():
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 80:
                S["camara"] = "La cámara dejó de entregar imagen. Cerrá otras apps que la estén usando y reiniciá la demo."
                print("[web] camara no responde"); break
            time.sleep(0.03); continue
        fails = 0
        now = time.time()
        with LOCK:
            BUF.append({"t": now, "frame": frame, "pts": False, "open": None})
        if now - last_jpg >= 0.05:                   # ~20fps al browser
            small = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            # con VARIAS caras: marcar a quien estamos leyendo (sticky lock)
            bb = S.get("lock_bbox")
            if bb and S.get("n_caras", 0) > 1:
                h2, w2 = small.shape[:2]
                x0, y0, x1, y1 = int(bb[0]*w2), int(bb[1]*h2), int(bb[2]*w2), int(bb[3]*h2)
                cv2.rectangle(small, (x0, y0), (x1, y1), (151, 179, 127), 2)
                cv2.putText(small, "leyendo a esta persona", (x0, max(y0-8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (151, 179, 127), 1)
            ok2, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2: JPEG["data"] = jpg.tobytes()
            last_jpg = now
    cap.release()

CAL_VPROC = VideoProcess()

def cal_preparar(idx, t0, t1, texto):
    """Corta una toma y la deja en memoria hasta que la persona la apruebe."""
    fin = time.time() + 2.5                       # esperar a que el hilo de landmarks alcance t1
    while time.time() < fin:
        with LOCK:
            pend = [e for e in BUF if t0 <= e["t"] <= t1 and e["pts"] is False]
        if not pend: break
        time.sleep(0.05)
    with LOCK:
        seg = [e for e in BUF if t0 <= e["t"] <= t1 and e["pts"] is not False]
    # No rechazamos tomas por cantidad de frames: la revisión humana decide si
    # sirve. Con menos de dos no hay intervalo temporal para procesar el clip.
    if len(seg) < 2:
        return False, "no llegué a capturar imagen; probá de nuevo", None
    dur = seg[-1]["t"] - seg[0]["t"]; fps = (len(seg)-1)/max(dur, 1e-3)
    pts = [e["pts"] for e in seg]
    if sum(1 for p in pts if p is not None)/len(pts) < 0.6:
        return False, "cara no detectada de forma estable — poné la cara de frente", None
    rgb = [np.ascontiguousarray(cv2.cvtColor(e["frame"], cv2.COLOR_BGR2RGB)) for e in seg]
    seq = CAL_VPROC(rgb, pts)
    if seq is None or len(seq) == 0:
        return False, "no pude recortar la boca — probá de nuevo", None
    arr = np.asarray(remuestrear_a_25fps([seq[i] for i in range(len(seq))], fps), dtype=np.uint8)
    tira = np.concatenate([arr[k] for k in np.linspace(0, len(arr)-1, 7).astype(int)], axis=1)
    ok, jpg = cv2.imencode(".jpg", tira, [cv2.IMWRITE_JPEG_QUALITY, 84])
    pendiente = {"frase_id": idx, "texto": texto, "rois": arr, "frames": int(len(arr)),
                 "duracion": round(dur, 2), "ratio": round(sum(p is not None for p in pts)/len(pts), 3)}
    return True, "toma lista para revisar", (pendiente, jpg.tobytes() if ok else None)

def estado_entrenamiento():
    with CAL_LOCK:
        persona = CAL["persona"]
    disponible = bool(persona) and os.path.isfile(os.path.join(MODELOS_PERSONAL, persona + ".pth"))
    with TRAIN_LOCK:
        activo = TRAIN["proc"] is not None and TRAIN["proc"].poll() is None
        return {"activo": activo, "fase": TRAIN["fase"], "salida": TRAIN["salida"][-16:], "error": TRAIN["error"],
                "modelo_disponible": disponible, "modelo_activo": S["config"].get("modelo", "base")}

def ruta_bash(ruta):
    """Convierte una ruta absoluta de Windows al formato que entiende Git Bash."""
    absoluta = os.path.abspath(os.path.expanduser(ruta))
    unidad, resto = os.path.splitdrive(absoluta)
    if unidad:
        return f"/{unidad[0].lower()}{resto.replace(os.sep, '/') }"
    return absoluta.replace(os.sep, "/")

def hilo_entrenamiento(persona):
    script = os.path.join(REPO, "personalization", "calibracion", "calibrar_entrenar.sh")
    try:
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        bash = git_bash if os.path.isfile(git_bash) else (os.environ.get("BASH") or shutil.which("bash") or "bash")
        env = dict(os.environ, VSR_PERSONAL_DIR=ruta_bash(PERSONAL_ROOT),
                   VSR_CAL_PY=ruta_bash(sys.executable))
        proc = subprocess.Popen([bash, ruta_bash(script), persona], cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        with TRAIN_LOCK: TRAIN.update(proc=proc, fase="preparando")
        for linea in proc.stdout:
            linea = linea.rstrip()
            with TRAIN_LOCK:
                TRAIN["salida"].append(linea)
                bajo = linea.lower()
                if "== 2/4" in linea: TRAIN["fase"] = "subiendo"
                elif "== 3/4" in linea: TRAIN["fase"] = "buscando_gpu"
                elif "== 4/4" in linea: TRAIN["fase"] = "entrenando"
                elif "listo" in bajo: TRAIN["fase"] = "listo"
        if proc.wait() != 0:
            with TRAIN_LOCK:
                salida = "\n".join(TRAIN["salida"])
                if "storage.objects.create" in salida:
                    error = ("No hay permiso para subir las tomas al bucket privado. "
                             "Hace falta roles/storage.objectAdmin sobre labios-argentos-vsr-dataset "
                             "para la cuenta activa.")
                else:
                    error = "El entrenamiento no pudo completarse. Revisá el detalle de abajo."
                TRAIN.update(fase="error", error=error)
        elif TRAIN["fase"] != "listo":
            with TRAIN_LOCK: TRAIN["fase"] = "listo"
    except Exception as e:
        with TRAIN_LOCK: TRAIN.update(fase="error", error=str(e))

def iniciar_entrenamiento(persona):
    if SESIONES.estado(persona)["hechas"] < 20:
        return False, "Necesitás al menos 20 tomas guardadas."
    with TRAIN_LOCK:
        if TRAIN["proc"] is not None and TRAIN["proc"].poll() is None:
            return False, "Ya hay un entrenamiento en curso."
        TRAIN.update(fase="preparando", salida=[], error="")
    threading.Thread(target=hilo_entrenamiento, args=(persona,), daemon=True).start()
    return True, "Preparando el entrenamiento."

def activar_modelo(persona, usar_base=False):
    """Recarga inferencia sin perder las tomas de la sesión."""
    ckpt = os.path.join(MODELOS_PERSONAL, persona + ".pth")
    if not usar_base and not os.path.isfile(ckpt):
        return False, "Todavía no hay un modelo personal terminado."
    try:
        with SRV["lock"]:
            if usar_base: os.environ.pop("VSR_CKPT", None)
            else: os.environ["VSR_CKPT"] = ckpt
            if SRV.get("proc"): SRV["proc"].terminate()
            SRV["proc"], cfg = lanzar_servidor(bool(S["config"].get("qwen")))
            cfg["max_seg"] = S["config"].get("max_seg"); cfg["run"] = RUN_ID
            S["config"] = cfg
        return True, "Modelo base activo." if usar_base else "Tu modelo personal está activo."
    except Exception as e:
        return False, str(e)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try: return json.loads(self.rfile.read(n).decode()) if n else {}
        except Exception: return {}
    def do_GET(self):
        if self.path == "/calibrar":
            body = open(os.path.join(os.path.dirname(HTML), "calibrar.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
        if self.path == "/modelo":
            body = open(HTML_MODELO, "rb").read(); self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path == "/cal/preview":
            with CAL_LOCK: d = CAL.get("preview")
            if not d:
                self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(d))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(d); return
        if self.path == "/cal/entrenamiento":
            self._json(200, estado_entrenamiento()); return
        if self.path.startswith("/strip/"):
            try:
                d = STRIPS[int(self.path.rsplit("/", 1)[1])]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(d)))
                # no-store: los ids de strip se reinician con el server; con cache el
                # navegador muestra bocas de una sesion anterior (de otra persona).
                self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(d)
            except Exception:
                self.send_response(404); self.end_headers()
            return
        if self.path == "/cal/estado":
            with CAL_LOCK:
                estado = SESIONES.estado(CAL["persona"]) if CAL["persona"] else {"hechas": 0, "siguiente": 0, "nivel": {}}
                modelo = os.path.join(MODELOS_PERSONAL, CAL["persona"] + ".pth") if CAL["persona"] else ""
                self._json(200, {"activo": CAL["activo"], "persona": CAL["persona"], **estado,
                                 "sugeridas": CAL_N, "total": len(CAL_PROMPTS), "niveles": NIVELES,
                                 "frases": CAL_PROMPTS, "grabando": CAL["rec_t0"] is not None,
                                 "pendiente": CAL["pendiente"] is not None, "modelo_disponible": os.path.isfile(modelo),
                                 "msg": CAL["msg"]})
            return
        if self.path == "/":
            body = open(HTML, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not STOP.is_set():
                    d = JPEG["data"]
                    if d:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(d)}\r\n\r\n".encode() + d + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while not STOP.is_set():
                    payload = json.dumps({k: S[k] for k in
                                          ("fase", "mov", "thr", "busy", "segmentos", "config", "n_caras")},
                                         ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode()); self.wfile.flush()
                    time.sleep(0.15)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path == "/qwen":                       # toggle del corrector en runtime
            on = bool(self._body().get("on"))
            try:
                with SRV["lock"]:
                    SRV["proc"].stdin.write(f"::qwen {int(on)}\n"); SRV["proc"].stdin.flush()
                    resp = SRV["proc"].stdout.readline().strip()
                ok = resp.startswith("::ok")
                # si el server rechaza el ON (Ollama caido) tambien deja qwen apagado de su lado
                S["config"]["qwen"] = on if ok else False
                self._json(200, {"ok": ok, "qwen": S["config"]["qwen"],
                                 "msg": "" if ok else resp.removeprefix("::err").strip()})
            except Exception as e:
                self._json(500, {"ok": False, "msg": str(e)})
        elif self.path == "/clear":
            S["segmentos"] = []
            self.send_response(204); self.end_headers()
        elif self.path == "/feedback":                 # correccion humana de una prediccion
            # Guarda pares prediccion→correccion en JSONL LOCAL (data/feedback/, gitignored).
            # No sale de la maquina: es materia prima para fine-tune personal / rescorer.
            b = self._body()
            try:
                idx = int(b.get("idx", -1))
                seg = S["segmentos"][idx]
            except (ValueError, IndexError):
                self._json(400, {"ok": False, "msg": "segmento invalido"}); return
            corregido = str(b.get("texto", "")).strip()
            if not corregido:
                self._json(400, {"ok": False, "msg": "texto vacio"}); return
            fila = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "texto_predicho": seg.get("texto", ""),
                    "texto_corregido": corregido,
                    "config": dict(S.get("config", {})),      # modelo/beam/qwen del CONFIG del server
                    "clip": {"idx": idx, "strip": seg.get("strip", -1),
                             "dur_s": seg.get("dur"), "infer_s": seg.get("infer")},
                    "consentimiento": "accion_explicita_local",  # la persona apreto guardar; solo local
                    "modo": "privado"}                            # sin envio externo (no hay codigo que suba esto)
            fdir = os.path.join(REPO, "data", "feedback"); os.makedirs(fdir, exist_ok=True)
            with open(os.path.join(fdir, "feedback.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
            seg["corregido"] = corregido               # la UI muestra que ya se corrigio
            self._json(200, {"ok": True})
        elif self.path == "/cal/entrar":
            b = self._body()
            persona = nombre_persona(b.get("persona"))
            estado = SESIONES.estado(persona)
            with CAL_LOCK:
                CAL.update(activo=True, persona=persona, dir=estado["carpeta"], rec_t0=None,
                           pendiente=None, preview=None, msg=f"Listo, {persona}.")
            self._json(200, {"ok": True, **estado})
        elif self.path == "/cal/salir":
            with CAL_LOCK:
                CAL.update(activo=False, rec_t0=None, msg="")
            self._json(200, {"ok": True})
        elif self.path == "/cal/rec":
            with CAL_LOCK:
                if not CAL["activo"]:
                    self._json(400, {"ok": False, "msg": "Primero ingresá tu nombre."}); return
                if CAL["pendiente"] is not None:
                    self._json(400, {"ok": False, "msg": "Primero elegí qué hacer con la toma anterior."}); return
                CAL["rec_t0"] = time.time(); CAL["msg"] = "grabando..."
            self._json(200, {"ok": True})
        elif self.path == "/cal/corte":
            b = self._body()
            with CAL_LOCK:
                t0 = CAL["rec_t0"]; CAL["rec_t0"] = None
                persona = CAL["persona"]
            if t0 is None:
                self._json(400, {"ok": False, "msg": "no estaba grabando"}); return
            estado = SESIONES.estado(persona, int(b.get("i", 0)))
            idx = estado["siguiente"] if estado["siguiente"] is not None else int(b.get("i", 0))
            if idx < 0 or idx >= len(CAL_PROMPTS):
                self._json(400, {"ok": False, "msg": "No quedan frases disponibles."}); return
            ok, msg, resultado = cal_preparar(idx, t0, time.time(), CAL_PROMPTS[idx])
            with CAL_LOCK:
                CAL["msg"] = msg
                if ok:
                    CAL["pendiente"], CAL["preview"] = resultado
            if not ok:
                SESIONES.evento(persona, "fallida", idx, motivo=msg)
                self._json(200, {"ok": False, "msg": msg}); return
            pendiente, _preview = resultado
            self._json(200, {"ok": True, "msg": msg, "duracion_s": pendiente["duracion"],
                             "ratio_deteccion": pendiente["ratio"], "n_frames": pendiente["frames"]})
        elif self.path == "/cal/aceptar":
            with CAL_LOCK:
                persona, pendiente = CAL["persona"], CAL["pendiente"]
                CAL["pendiente"] = None; CAL["preview"] = None
            if not pendiente:
                self._json(400, {"ok": False, "msg": "No hay una toma para guardar."}); return
            estado = SESIONES.aceptar(persona, pendiente)
            encolar_subida(SESIONES.carpeta(persona), int(pendiente["frase_id"]), persona)
            self._json(200, {"ok": True, "msg": "Toma guardada.", **estado})
        elif self.path == "/cal/descartar":
            b = self._body()
            with CAL_LOCK:
                persona, pendiente = CAL["persona"], CAL["pendiente"]
                CAL["pendiente"] = None; CAL["preview"] = None
            frase_id = (pendiente or {}).get("frase_id", SESIONES.estado(persona).get("siguiente", 0))
            if b.get("seguir"):
                estado = SESIONES.omitir(persona, frase_id)
            else:
                SESIONES.evento(persona, "reintento", frase_id)
                estado = SESIONES.estado(persona, frase_id)
            self._json(200, {"ok": True, "msg": "Toma descartada.", **estado})
        elif self.path == "/cal/entrenar":
            with CAL_LOCK: persona = CAL["persona"]
            ok, msg = iniciar_entrenamiento(persona)
            self._json(200 if ok else 400, {"ok": ok, "msg": msg})
        elif self.path == "/cal/activar":
            with CAL_LOCK: persona = CAL["persona"]
            ok, msg = activar_modelo(persona); self._json(200 if ok else 400, {"ok": ok, "msg": msg})
        elif self.path == "/cal/base":
            ok, msg = activar_modelo("", usar_base=True); self._json(200 if ok else 400, {"ok": ok, "msg": msg})
        else:
            self.send_response(404); self.end_headers()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8551)
    ap.add_argument("--max-seg", type=float, default=4.0)
    ap.add_argument("--pause", type=float, default=0.45)
    ap.add_argument("--sens", type=float, default=3.0)
    ap.add_argument("--qwen", action="store_true")
    ap.add_argument("--ckpt", default="", help="checkpoint personalizado (modelos/personal/<n>.pth)")
    ap.add_argument("--no-open", action="store_true", help="no abrir el browser solo")
    args = ap.parse_args()

    if args.ckpt:
        os.environ["VSR_CKPT"] = os.path.expanduser(args.ckpt)
    SRV["proc"], cfg = lanzar_servidor(args.qwen)
    cfg["max_seg"] = args.max_seg
    cfg["run"] = RUN_ID
    S["config"] = cfg

    for fn in (hilo_camara, hilo_landmarks, hilo_subida_clips):
        threading.Thread(target=fn, daemon=True).start()
    threading.Thread(target=hilo_segmentador, args=(args,), daemon=True).start()

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    url = f"http://localhost:{args.port}"
    print(f"[web] UI en {url}  (Ctrl-C para salir; quedate callado ~2s al principio: calibra el VAD)")
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    STOP.set()
    if S["segmentos"]:
        print("\n[web] transcript final:\n  " + " ".join(g["texto"] for g in S["segmentos"]))
    try:
        SRV["proc"].stdin.close(); SRV["proc"].terminate()
    except Exception:
        pass

if __name__ == "__main__":
    main()
