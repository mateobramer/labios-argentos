#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demo push-to-talk de VSR rioplatense.
Abris la camara, apretas ESPACIO para empezar a grabar, hablas (moviendo la boca),
apretas ESPACIO de nuevo para cortar, y el modelo (ViSpeR) te tira la transcripcion.

Pipeline: webcam -> (al cortar) MediaPipe 478 landmarks -> 4 puntos estables ->
warp a cara media -> 96x96 gris 25fps (MISMO crop que el entrenamiento) -> npz ->
servidor de inferencia ViSpeR (proceso aparte en el env `visper`) -> texto.

Corre en el env `ptt`:
  ~/miniconda3/envs/ptt/bin/python demo_ptt.py

Teclas:  ESPACIO = empezar/cortar grabacion    |    q = salir
"""
import os, sys, time, subprocess, tempfile
import cv2, numpy as np

REPO = os.path.expanduser("~/Desktop/labios-argentos")
sys.path.insert(0, REPO)
from visual_preprocessing.src.preprocesar import (
    crear_landmarker, detectar_landmarks, cuatro_puntos, remuestrear_a_25fps,
)
from visual_preprocessing.src.video_process import VideoProcess

VISPER_PY = os.path.expanduser("~/miniconda3/envs/visper/bin/python")
INFER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_server.py")
NPZ = os.path.join(tempfile.gettempdir(), "ptt_clip.npz")
MIN_FRAMES = 12   # ~0.5s: menos que esto no vale la pena

def lanzar_servidor():
    print("[demo] cargando el modelo ViSpeR... la PRIMERA vez tarda ~20-40s. NO cortes (Ctrl-C).", flush=True)
    print("[demo] (cuando diga 'servidor listo' se abre la camara)", flush=True)
    p = subprocess.Popen([VISPER_PY, INFER], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:               # espera el READY
        if line.strip() == "READY":
            break
    print("[demo] servidor listo.", flush=True)
    return p

def transcribir(server, frames_bgr, fps, landmarker, vproc):
    """frames BGR crudos -> crop 96x96 -> npz -> servidor -> texto."""
    frames_rgb, puntos, det = [], [], 0
    for f in frames_bgr:
        rgb = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        frames_rgb.append(rgb)
        lm = detectar_landmarks(rgb, landmarker)
        if lm is not None:
            puntos.append(cuatro_puntos(lm, rgb.shape[1], rgb.shape[0])); det += 1
        else:
            puntos.append(None)
    if det / max(len(frames_rgb), 1) < 0.5:
        return "(no detecte una cara estable, proba de nuevo mas de frente)"
    seq = vproc(frames_rgb, puntos)
    if seq is None or len(seq) == 0:
        return "(no pude recortar la boca)"
    seq = remuestrear_a_25fps([seq[i] for i in range(len(seq))], fps)
    np.savez_compressed(NPZ, rois=np.asarray(seq, dtype=np.uint8))
    server.stdin.write(NPZ + "\n"); server.stdin.flush()
    out = server.stdout.readline().strip()
    return out if not out.startswith("__ERROR__") else f"(error inferencia: {out})"

def wrap(txt, n=48):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur: lines.append(cur)
    return lines or [""]

def main():
    landmarker = crear_landmarker()
    vproc = VideoProcess()
    server = lanzar_servidor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[demo] ERROR: no pude abrir la camara (index 0)."); return
    recording, frames, t0, last, last_toggle = False, [], 0.0, "Apreta ESPACIO y habla.", 0.0
    print("[demo] camara abierta. ESPACIO = grabar/cortar, q = salir.")

    while True:
        ok, frame = cap.read()
        if not ok: break
        view = frame.copy()
        h = view.shape[0]
        if recording:
            frames.append(frame.copy())
            cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(view, f"REC {len(frames)}f", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(view, "ESPACIO=grabar  q=salir", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, ln in enumerate(wrap(last)):
            cv2.putText(view, ln, (20, h - 20 - 22*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("VSR push-to-talk (rioplatense)", view)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if key == ord('q'):
            break
        # DEBOUNCE: un tap de ESPACIO se ve en varios frames; ignoramos toggles < 0.6s
        if key == ord(' ') and (now - last_toggle) > 0.6:
            last_toggle = now
            if not recording:
                recording, frames, t0 = True, [], now
                last = "grabando... (ESPACIO para cortar)"
            else:
                recording = False
                dt = max(now - t0, 1e-3)
                if len(frames) < MIN_FRAMES:
                    last = f"muy corto ({len(frames)}f). Apreta ESPACIO, habla 2-4s, ESPACIO de nuevo."
                    print(f"[demo] descartado: {len(frames)} frames")
                else:
                    fps = len(frames) / dt
                    proc = frame.copy()   # pinta 'procesando' antes de bloquear ~6s
                    cv2.putText(proc, "procesando...", (20, proc.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("VSR push-to-talk (rioplatense)", proc); cv2.waitKey(1)
                    last = transcribir(server, frames, fps, landmarker, vproc)
                    print(f"[demo] ({len(frames)}f @ {fps:.1f}fps) -> {last}")

    cap.release(); cv2.destroyAllWindows()
    try:
        server.stdin.close(); server.terminate()
    except Exception:
        pass

if __name__ == "__main__":
    main()
