#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demo VSR "casi tiempo real" con VAD VISUAL + transcript acumulado (stitching).

En vez de ventanas a intervalo fijo, corta los segmentos en las PAUSAS de la boca
(movimiento de labios medido con MediaPipe: apertura labio 13-14 normalizada por el
alto de cara 10-152, sobre landmarks que se calculan EN VIVO durante la captura).
Cada segmento se infiere (ViSpeR beam3, server aparte) y su texto se APPENDEA a un
transcript acumulado que crece en pantalla (segmentos disjuntos -> sin costuras;
si un segmento llega al tope de largo se corta forzado y se dedup-ea el borde).

Al arrancar: quedate CALLADO ~2s (calibra el ruido de base del VAD).

OJO: ViSpeR es offline/bidireccional -> esto sigue siendo una aproximacion
(latencia ~ fin-de-frase + inferencia ~1.5s). Ver experiments/09 y realtime-vsr-plan.

Corre en el env `ptt` (spawnea el server en el env `visper`):
  ~/miniconda3/envs/ptt/bin/python demo_stream.py [--max-seg 4.0] [--pause 0.45] [--sens 3.0] [--qwen]

Teclas:  c = limpiar transcript   q = salir
"""
import os, sys, time, threading, subprocess, tempfile, argparse, collections, statistics
import cv2, numpy as np

# Raiz del repo derivada de este archivo; LABIOS_REPO la pisa (ver .env.example).
REPO = os.environ.get("LABIOS_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from visual_preprocessing.src.preprocesar import (
    crear_landmarker, detectar_landmarks, cuatro_puntos, remuestrear_a_25fps)
from visual_preprocessing.src.video_process import VideoProcess

VISPER_PY = os.path.expanduser(os.environ.get("VISPER_PY", "~/miniconda3/envs/visper/bin/python"))
INFER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_server.py")
MIN_SEG_S = 0.7      # segmentos mas cortos que esto se descartan (no vale inferir)
PREROLL_S = 0.25     # arranque del segmento un toque antes de detectar habla
MOV_WIN_S = 0.35     # ventana del movimiento (std de apertura)
CALIB_S = 2.0        # silencio inicial para calibrar el VAD

def lanzar_servidor(use_qwen):
    env = dict(os.environ, VSR_BEAM="3")
    if use_qwen: env["VSR_QWEN"] = "1"
    print("[stream] cargando ViSpeR (primera vez ~20-40s, NO cortes)...", flush=True)
    p = subprocess.Popen([VISPER_PY, INFER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        if line.strip() == "READY": break
    print("[stream] server listo.", flush=True)
    return p

def wrap(txt, n=56):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur)+len(w)+1 > n: lines.append(cur); cur = w
        else: cur = (cur+" "+w).strip()
    if cur: lines.append(cur)
    return lines or [""]

def apertura(lm):
    """Apertura de boca normalizada por alto de cara (landmarks normalizados)."""
    boca = abs(lm[14].y - lm[13].y)          # labio interno inferior - superior
    cara = abs(lm[152].y - lm[10].y) + 1e-6  # menton - frente
    return boca / cara

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seg", type=float, default=4.0, help="tope de segundos por segmento (corte forzado)")
    ap.add_argument("--pause", type=float, default=0.45, help="segundos de boca quieta que cierran el segmento")
    ap.add_argument("--sens", type=float, default=3.0, help="umbral = sens x ruido base (mas bajo = mas sensible)")
    ap.add_argument("--qwen", action="store_true", help="corregir con qwen (mas lento, ~+1.2s)")
    args = ap.parse_args()

    server = lanzar_servidor(args.qwen)
    npz_path = os.path.join(tempfile.gettempdir(), "stream_seg.npz")

    # buffer compartido: cada entrada {t, frame, pts, open} (pts/open los llena el hilo de landmarks)
    buf = collections.deque()
    lock = threading.Lock()
    state = {"fase": "calibrando (quedate callado)...", "mov": 0.0, "thr": None,
             "transcript": [], "parcial": "", "busy": False, "ultimo_forzado": False}
    stop = threading.Event()

    def hilo_landmarks():
        """Anota pts + apertura en las entradas nuevas del buffer (tiempo real)."""
        lmk = crear_landmarker()
        i = 0
        while not stop.is_set():
            with lock:
                pend = [e for e in buf if e["pts"] is False][:4]
            if not pend:
                time.sleep(0.01); continue
            for e in pend:
                rgb = np.ascontiguousarray(cv2.cvtColor(e["frame"], cv2.COLOR_BGR2RGB))
                lm = detectar_landmarks(rgb, lmk)
                if lm is not None:
                    e["pts"] = cuatro_puntos(lm, rgb.shape[1], rgb.shape[0])
                    e["open"] = apertura(lm)
                else:
                    e["pts"] = None; e["open"] = None
            i += len(pend)

    def hilo_segmentador():
        """VAD sobre la serie de apertura -> cierra segmentos -> infiere -> appendea."""
        vproc = VideoProcess()
        aper = collections.deque()   # (t, open) ya anotados
        base, t_ini = [], time.time()
        thr = None
        hablando, seg_t0, quieto_desde = False, 0.0, None
        consumido_hasta = 0.0        # timestamp hasta donde ya procesamos la serie

        def movimiento(t_now):
            vals = [o for (t, o) in aper if t >= t_now - MOV_WIN_S and o is not None]
            return statistics.pstdev(vals) if len(vals) >= 3 else 0.0

        def inferir(t_a, t_b, forzado):
            with lock:
                seg = [e for e in buf if t_a <= e["t"] <= t_b and e["pts"] is not False]
            if not seg or (seg[-1]["t"] - seg[0]["t"]) < MIN_SEG_S: return
            fps = (len(seg)-1) / max(seg[-1]["t"] - seg[0]["t"], 1e-3)
            rgb = [np.ascontiguousarray(cv2.cvtColor(e["frame"], cv2.COLOR_BGR2RGB)) for e in seg]
            pts = [e["pts"] for e in seg]
            det = sum(1 for p in pts if p is not None) / len(pts)
            if det < 0.5:
                state["parcial"] = ""; return
            state["busy"] = True; state["parcial"] = "…"
            t0 = time.time()
            try:
                out = ""
                seq = vproc(rgb, pts)
                if seq is not None and len(seq) > 0:
                    arr = np.asarray(remuestrear_a_25fps([seq[i] for i in range(len(seq))], fps), dtype=np.uint8)
                    np.savez_compressed(npz_path, rois=arr)
                    server.stdin.write(npz_path + "\n"); server.stdin.flush()
                    out = server.stdout.readline().strip()
            except Exception as e:
                out = f"__ERROR__ {e}"
            dt = time.time() - t0
            if out and not out.startswith("__ERROR__"):
                palabras = out.split()
                # dedup de borde: si el segmento anterior se corto forzado y repite la palabra
                if state["ultimo_forzado"] and state["transcript"]:
                    prev = state["transcript"][-1].split()
                    if prev and palabras and prev[-1] == palabras[0]:
                        palabras = palabras[1:]
                if palabras:
                    state["transcript"].append(" ".join(palabras))
            state["ultimo_forzado"] = forzado
            state["parcial"] = ""; state["busy"] = False
            state["fase"] = f"({seg[-1]['t']-seg[0]['t']:.1f}s de habla -> infer {dt:.1f}s)"

        while not stop.is_set():
            with lock:
                nuevos = [(e["t"], e["open"]) for e in buf
                          if e["pts"] is not False and e["t"] > consumido_hasta]
            if not nuevos:
                time.sleep(0.015); continue
            for (t, o) in nuevos:
                consumido_hasta = t
                aper.append((t, o))
                while aper and aper[0][0] < t - max(MOV_WIN_S, 1.0):
                    aper.popleft()
                # --- calibracion inicial (silencio) ---
                if thr is None:
                    if o is not None: base.append(o)
                    if t - t_ini >= CALIB_S and len(base) >= 10:
                        ruido = statistics.pstdev(base[-40:])
                        thr = max(args.sens * ruido, 0.004)
                        state["thr"] = thr
                        state["fase"] = "listo: habla cuando quieras"
                    continue
                mov = movimiento(t); state["mov"] = mov
                # --- maquina de estados ---
                if not hablando:
                    if mov > thr:
                        hablando, seg_t0, quieto_desde = True, t - PREROLL_S, None
                        state["fase"] = "HABLANDO"
                else:
                    if mov <= thr:
                        quieto_desde = quieto_desde or t
                        if t - quieto_desde >= args.pause:      # pausa -> cierre natural
                            hablando = False; state["fase"] = "pausa detectada"
                            inferir(seg_t0, quieto_desde + 0.1, forzado=False)
                    else:
                        quieto_desde = None
                    if hablando and t - seg_t0 >= args.max_seg:  # tope -> corte forzado
                        inferir(seg_t0, t, forzado=True)
                        seg_t0 = t   # el proximo segmento sigue de inmediato
            # podar buffer: todo lo anterior al segmento activo ya no hace falta
            corte = (seg_t0 if hablando else consumido_hasta) - PREROLL_S - 0.5
            with lock:
                while buf and buf[0]["t"] < corte:
                    buf.popleft()

    th1 = threading.Thread(target=hilo_landmarks, daemon=True); th1.start()
    th2 = threading.Thread(target=hilo_segmentador, daemon=True); th2.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[stream] ERROR: no abre la camara"); stop.set(); return
    print("[stream] calentando camara...", flush=True)
    for _ in range(40):
        okw, _f = cap.read()
        if okw: break
        time.sleep(0.2)
    print(f"[stream] listo. VAD: pause={args.pause}s max-seg={args.max_seg}s sens={args.sens} "
          f"qwen={'ON' if args.qwen else 'off'}. QUEDATE CALLADO ~2s (calibra). c=limpiar q=salir")

    fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 80: print("[stream] la camara dejo de responder"); break
            time.sleep(0.03); continue
        fails = 0
        with lock:
            buf.append({"t": time.time(), "frame": frame.copy(), "pts": False, "open": None})

        view = frame.copy(); h, w = view.shape[:2]
        texto = " ".join(state["transcript"][-12:]) + (" " + state["parcial"] if state["parcial"] else "")
        lines = wrap(texto)[-4:] if texto.strip() else ["(hablá; el texto se acumula acá)"]
        band = 26 * (len(lines) + 1) + 16
        cv2.rectangle(view, (0, h - band), (w, h), (0, 0, 0), -1)
        dot = (0, 165, 255) if state["busy"] else ((0, 200, 0) if state["thr"] else (0, 200, 255))
        cv2.circle(view, (22, h - band + 20), 9, dot, -1)
        mv = state["mov"]; thr = state["thr"] or 1.0
        barra = min(int(60 * mv / (thr * 2 + 1e-9)), 60)
        cv2.rectangle(view, (40, h - band + 12), (40 + barra, h - band + 24), (0, 200, 0), -1)
        cv2.rectangle(view, (40, h - band + 12), (100, h - band + 24), (90, 90, 90), 1)
        cv2.putText(view, state["fase"], (112, h - band + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, ln in enumerate(lines):
            cv2.putText(view, ln, (15, h - band + 52 + i*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.imshow("VSR streaming (VAD visual, rioplatense)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'): state["transcript"] = []

    stop.set(); cap.release(); cv2.destroyAllWindows()
    if state["transcript"]:
        print("\n[stream] transcript final:\n  " + " ".join(state["transcript"]))
    try:
        server.stdin.close(); server.terminate()
    except Exception:
        pass

if __name__ == "__main__":
    main()
