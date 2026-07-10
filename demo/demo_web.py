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
import os, re, sys, time, json, threading, subprocess, tempfile, argparse, collections, statistics, webbrowser
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
MIN_SEG_S, PREROLL_S, MOV_WIN_S, CALIB_S = 0.7, 0.25, 0.35, 2.0

# ---- estado compartido (los handlers http lo leen) ----
S = {"fase": "calibrando", "mov": 0.0, "thr": None, "busy": False,
     "segmentos": [], "config": {}, "ultimo_forzado": False,
     "n_caras": 0, "lock_bbox": None}
JPEG = {"data": None}
STRIPS = []          # tiras "lo que ve el modelo": jpg de 7 bocas 96x96 por segmento
BUF = collections.deque()
LOCK = threading.Lock()
STOP = threading.Event()
SRV = {"proc": None, "lock": threading.Lock()}

# ---- modo calibracion (grabar frases para adaptar el modelo a una persona) ----
from personalization.build_testset import PROMPTS as CAL_PROMPTS  # mismas 100 frases del self-test (REPO ya esta en sys.path)
CAL_N = 40                                                 # frases sugeridas (30 ya captura ~80%)
CAL = {"activo": False, "persona": "", "dir": None, "rec_t0": None, "hechas": 0, "msg": ""}
CAL_LOCK = threading.Lock()

def cal_dir(persona):
    d = os.path.expanduser(f"~/vsr_personal/{persona}")
    os.makedirs(d, exist_ok=True); return d

def cal_contar(d):
    import glob as _g
    return len(_g.glob(os.path.join(d, "clip_*.npz")))

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
    # (beam5 vs beam3 = mismo WER, +0.15s; ver experiments/09)
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[web] ERROR: no abre la camara"); STOP.set(); return
    for _ in range(40):                              # warmup macOS
        ok, _f = cap.read()
        if ok: break
        time.sleep(0.2)
    print("[web] camara ok.", flush=True)
    last_jpg = 0.0
    fails = 0
    while not STOP.is_set():
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 80: print("[web] camara no responde"); break
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

def cal_guardar(idx, t0, t1, texto):
    """Corta [t0,t1] del buffer, croppea con los landmarks ya computados y guarda npz+txt."""
    fin = time.time() + 2.5                       # esperar a que el hilo de landmarks alcance t1
    while time.time() < fin:
        with LOCK:
            pend = [e for e in BUF if t0 <= e["t"] <= t1 and e["pts"] is False]
        if not pend: break
        time.sleep(0.05)
    with LOCK:
        seg = [e for e in BUF if t0 <= e["t"] <= t1 and e["pts"] is not False]
    if len(seg) < 20:
        return False, f"muy corto ({len(seg)} frames) — mantené apretado mientras leés"
    dur = seg[-1]["t"] - seg[0]["t"]; fps = (len(seg)-1)/max(dur, 1e-3)
    pts = [e["pts"] for e in seg]
    if sum(1 for p in pts if p is not None)/len(pts) < 0.6:
        return False, "cara no detectada de forma estable — poné la cara de frente"
    rgb = [np.ascontiguousarray(cv2.cvtColor(e["frame"], cv2.COLOR_BGR2RGB)) for e in seg]
    seq = CAL_VPROC(rgb, pts)
    if seq is None or len(seq) == 0:
        return False, "no pude recortar la boca — probá de nuevo"
    arr = np.asarray(remuestrear_a_25fps([seq[i] for i in range(len(seq))], fps), dtype=np.uint8)
    base = os.path.join(CAL["dir"], f"clip_{idx:02d}")
    np.savez_compressed(base + ".npz", rois=arr)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(texto)
    return True, f"guardada ({arr.shape[0]} frames @25fps)"

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
        if self.path.startswith("/strip/"):
            try:
                d = STRIPS[int(self.path.rsplit("/", 1)[1])]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(d)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers(); self.wfile.write(d)
            except Exception:
                self.send_response(404); self.end_headers()
            return
        if self.path == "/cal/estado":
            with CAL_LOCK:
                hechas = cal_contar(CAL["dir"]) if CAL["dir"] else 0
                self._json(200, {"activo": CAL["activo"], "persona": CAL["persona"],
                                 "hechas": hechas, "sugeridas": CAL_N, "total": len(CAL_PROMPTS),
                                 "frases": CAL_PROMPTS, "grabando": CAL["rec_t0"] is not None,
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
                S["config"]["qwen"] = on
                self._json(200, {"ok": resp.startswith("::ok"), "qwen": on})
            except Exception as e:
                self._json(500, {"ok": False, "msg": str(e)})
        elif self.path == "/clear":
            S["segmentos"] = []
            self.send_response(204); self.end_headers()
        elif self.path == "/cal/entrar":
            b = self._body()
            persona = re.sub(r"[^a-z0-9_-]", "", str(b.get("persona", "")).lower().strip()) or "persona"
            modo = "contrib" if b.get("modo") == "contrib" else "calibrar"
            raiz = "~/vsr_contrib" if modo == "contrib" else "~/vsr_personal"
            d = os.path.expanduser(f"{raiz}/{persona}"); os.makedirs(d, exist_ok=True)
            with CAL_LOCK:
                CAL.update(activo=True, persona=persona, dir=d, modo=modo,
                           rec_t0=None, msg=f"listo, {persona}")
            self._json(200, {"ok": True, "persona": persona, "modo": modo,
                             "hechas": cal_contar(d)})
        elif self.path == "/cal/salir":
            with CAL_LOCK:
                CAL.update(activo=False, rec_t0=None, msg="")
            self._json(200, {"ok": True})
        elif self.path == "/cal/rec":
            with CAL_LOCK:
                CAL["rec_t0"] = time.time(); CAL["msg"] = "grabando..."
            self._json(200, {"ok": True})
        elif self.path == "/cal/corte":
            b = self._body()
            with CAL_LOCK:
                t0 = CAL["rec_t0"]; CAL["rec_t0"] = None
                modo = CAL.get("modo", "calibrar")
            if t0 is None:
                self._json(400, {"ok": False, "msg": "no estaba grabando"}); return
            if modo == "contrib":
                idx = cal_contar(CAL["dir"])                     # siguiente libre
                texto = norm_texto(str(b.get("texto", "")))
                if not texto:
                    self._json(200, {"ok": False, "msg": "escribí lo que dijiste antes de cortar"}); return
            else:
                idx = int(b.get("i", 0)); texto = CAL_PROMPTS[idx]
            ok, msg = cal_guardar(idx, t0, time.time(), texto)
            with CAL_LOCK: CAL["msg"] = msg
            self._json(200, {"ok": ok, "msg": msg, "hechas": cal_contar(CAL["dir"])})
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
    S["config"] = cfg

    for fn in (hilo_camara, hilo_landmarks):
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
