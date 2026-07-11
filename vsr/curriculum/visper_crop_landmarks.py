#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""FASE 0.b — cropper ViSpeR usando SUS landmarks (bypassa MediaPipe).

ViSpeR ya trae 5 puntos/frame (RetinaFace, normalizados 0-1) por segmento:
  [ojo0, ojo1, nariz, boca_izq, boca_der]  (ojo0.x < ojo1.x = ojo der/izq como la mean-face).
Los mapeamos a los 4 puntos que espera VideoProcess ([ojo der, ojo izq, nariz, centro boca])
y corremos el MISMO warp mean-face + crop 96x96 gris que el resto del pipeline. Sin re-detectar.

Uso:
  .venv-cleaning/bin/python vsr/curriculum/visper_crop_landmarks.py \
      --jsons <tedx.json> <wild.json> --raw-dir data/_visper_tmp --max-clips 6
Salida: data/processed/lip_rois/visper_<id>/clip_NNNN.npz (+ .txt) y un resumen de yield.
"""
import os, sys, json, argparse, glob
import numpy as np, cv2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "preprocessing", "src"))
from video_process import VideoProcess

SALIDA = os.path.join(RAIZ, "data", "processed", "lip_rois")
FPS_OUT = 25

def resample_25(frames, fps_src):
    if fps_src <= 0 or abs(fps_src - FPS_OUT) < 0.5 or len(frames) == 0:
        return frames
    dur = len(frames) / fps_src
    n_out = max(1, round(dur * FPS_OUT))
    idx = [min(len(frames) - 1, round(k * fps_src / FPS_OUT)) for k in range(n_out)]
    return [frames[i] for i in idx]

def leer_frames(cap, fps, start, end):
    f0 = int(round(start * fps)); f1 = int(round(end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(max(0, f1 - f0)):
        ok, fr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return frames

def cuatro_puntos(seg_lms, W, H):
    """seg_lms: lista (por frame) de 5 puntos normalizados -> lista de arrays 4x2 en pixeles."""
    out = []
    for p in seg_lms:
        a = np.asarray(p, dtype=np.float32)          # (5,2) normalizado
        a[:, 0] *= W; a[:, 1] *= H
        cuatro = np.vstack([a[0], a[1], a[2], (a[3] + a[4]) / 2.0])  # ojoD, ojoI, nariz, boca
        out.append(cuatro)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True)
    ap.add_argument("--raw-dir", default=os.path.join(RAIZ, "data", "_visper_tmp"))
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    meta = {}
    for j in args.jsons:
        meta.update(json.load(open(j, encoding="utf-8")))
    vids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(args.raw_dir, "*.mp4"))]
    vids = [v for v in vids if v in meta]
    print(f"[fase0.b] {len(vids)} videos crudos con metadata | jsons={len(meta)} ids")

    vproc = VideoProcess(mean_face_path="20words_mean_face.npy", convert_gray=True)
    tot_ok = tot_fail = tot_frames = 0
    for vi, vid in enumerate(vids, 1):
        raw = os.path.join(args.raw_dir, vid + ".mp4")
        cap = cv2.VideoCapture(raw)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        segs = meta[vid][:args.max_clips] if args.max_clips else meta[vid]
        dst = os.path.join(SALIDA, f"visper_{vid}"); os.makedirs(dst, exist_ok=True)
        ok = fail = 0
        for k, s in enumerate(segs, 1):
            frames = leer_frames(cap, fps, s["start"], s["end"])
            lms = s["landmarks"]
            if not frames or not lms:
                fail += 1; continue
            # emparejar m frames con n landmarks por indice proporcional
            m, n = len(frames), len(lms)
            pares = [lms[min(n - 1, int(i * n / m))] for i in range(m)]
            pts = cuatro_puntos(pares, W, H)
            try:
                seq = vproc(frames, pts)   # (m,96,96) gris uint8
            except Exception as e:
                fail += 1; continue
            if seq is None or len(seq) == 0:
                fail += 1; continue
            seq = resample_25(list(seq), fps)
            arr = np.asarray(seq, dtype=np.uint8)
            base = os.path.join(dst, f"clip_{k:04d}")
            np.savez_compressed(base + ".npz", rois=arr)
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(str(s.get("label", "")).strip() + "\n")
            ok += 1; tot_frames += len(arr)
        cap.release()
        tot_ok += ok; tot_fail += fail
        print(f"  [{vi}/{len(vids)}] {vid} ({W}x{H}@{fps:.0f}fps): ok={ok} fail={fail}", flush=True)

    tot = tot_ok + tot_fail
    print(f"\n===== FASE 0.b (landmarks ViSpeR) =====")
    print(f"clips ok={tot_ok} fail={tot_fail} | yield={100*tot_ok/max(tot,1):.0f}% | horas_npz={tot_frames/FPS_OUT/3600:.2f}h")

if __name__ == "__main__":
    main()
