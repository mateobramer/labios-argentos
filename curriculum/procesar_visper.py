#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""FASE 1 — orquestador de procesamiento ViSpeR-es (descarga + crop via landmarks).

Recorre los video-ids de los JSON de ViSpeR (train), baja cada uno con yt-dlp, corta
sus segmentos y croppea con los landmarks de ViSpeR (mismo warp mean-face que ft05,
bypassa MediaPipe) -> data/processed/lip_rois/visper_<id>/clip_NNNN.npz (+ .txt).
Acumula hasta --target-hours. Reanudable, borra crudos tras croppear, respalda al bucket.

Uso (con el env pineado):
  .venv-cleaning/bin/python curriculum/procesar_visper.py --json-dir <dir> \
      --target-hours 50 [--max-videos N] [--bucket gs://...] [--sleep 2]
"""
import os, sys, json, glob, time, argparse, subprocess, random, shutil
import numpy as np, cv2, ijson

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "preprocessing", "src"))
from video_process import VideoProcess

SALIDA = os.path.join(RAIZ, "data", "processed", "lip_rois")
FPS_OUT = 25

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def descargar(vid, dst, sleep, hard_timeout=120):
    url = f"https://www.youtube.com/watch?v={vid}"
    try:
        r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "--socket-timeout", "20",
                            "--retries", "2", "--no-part", "-f", "18/bv*[height<=480]+ba/b[height<=480]",
                            "-o", dst, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=hard_timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -1  # descarga colgada -> matada por timeout duro, se saltea
    for p in (dst, dst + ".part"):   # limpiar parciales de una descarga cortada
        if os.path.exists(p) and rc != 0:
            try: os.remove(p)
            except OSError: pass
    if sleep:
        time.sleep(sleep)
    return rc == 0 and os.path.exists(dst)

def resample_25(frames, fps_src):
    if fps_src <= 0 or abs(fps_src - FPS_OUT) < 0.5 or not len(frames):
        return frames
    dur = len(frames) / fps_src
    n_out = max(1, round(dur * FPS_OUT))
    idx = [min(len(frames) - 1, round(k * fps_src / FPS_OUT)) for k in range(n_out)]
    return [frames[i] for i in idx]

def cuatro(seg_lms, W, H):
    out = []
    for p in seg_lms:
        a = np.asarray(p, dtype=np.float32); a[:, 0] *= W; a[:, 1] *= H
        out.append(np.vstack([a[0], a[1], a[2], (a[3] + a[4]) / 2.0]))
    return out

def crop_video(raw, segs, vproc, dst):
    cap = cv2.VideoCapture(raw)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(dst, exist_ok=True)
    ok = 0; frames_tot = 0
    for k, s in enumerate(segs, 1):
        base = os.path.join(dst, f"clip_{k:04d}")
        if os.path.exists(base + ".npz"):
            try: frames_tot += len(np.load(base + ".npz")["rois"]); ok += 1; continue
            except Exception: pass
        f0 = int(round(float(s["start"]) * fps)); f1 = int(round(float(s["end"]) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames = []
        for _ in range(max(0, f1 - f0)):
            r, fr = cap.read()
            if not r: break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        lms = s.get("landmarks") or []
        if not frames or not lms:
            continue
        m, n = len(frames), len(lms)
        pares = [lms[min(n - 1, int(i * n / m))] for i in range(m)]
        try:
            seq = vproc(frames, cuatro(pares, W, H))
        except Exception:
            continue
        if seq is None or not len(seq):
            continue
        arr = np.asarray(resample_25(list(seq), fps), dtype=np.uint8)
        np.savez_compressed(base + ".npz", rois=arr)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(str(s.get("label", "")).strip() + "\n")
        ok += 1; frames_tot += len(arr)
    cap.release()
    return ok, frames_tot / FPS_OUT / 3600.0  # clips ok, horas

def horas_existentes():
    """Reanudacion: suma horas de npz ya en disco."""
    tot = 0.0; vids = set()
    for f in glob.glob(os.path.join(SALIDA, "visper_*", "*.npz")):
        vids.add(os.path.basename(os.path.dirname(f)))
        try: tot += len(np.load(f)["rois"]) / FPS_OUT / 3600.0
        except Exception: pass
    return tot, vids

def backup(bucket):
    if not bucket: return
    try:
        subprocess.run(["gcloud", "storage", "rsync", "-r", SALIDA,
                        f"{bucket}/curriculum_visper/lip_rois"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
        log("backup al bucket OK")
    except Exception as e:
        log(f"backup fallo: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--target-hours", type=float, default=50.0)
    ap.add_argument("--max-videos", type=int, default=0, help="tope de videos (para test)")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--sleep", type=float, default=2.0, help="pausa anti rate-limit entre descargas")
    ap.add_argument("--backup-cada", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--raw-tmp", default=os.path.join(RAIZ, "data", "_visper_tmp"))
    args = ap.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.json_dir, "**", "*.json"), recursive=True))
    os.makedirs(args.raw_tmp, exist_ok=True); os.makedirs(SALIDA, exist_ok=True)
    vproc = VideoProcess(convert_gray=True)  # carga mean_face una sola vez

    def stream_videos():
        # streamea (video_id, segmentos) de a uno con ijson (RAM minima; JSON de 2GB c/u)
        for j in jsons:
            with open(j, "rb") as f:
                for vid, segs in ijson.kvitems(f, "", use_float=True):
                    yield vid, segs

    hrs, done = horas_existentes()
    log(f"START | chunks={len(jsons)} | ya en disco: {hrs:.2f}h en {len(done)} videos | target={args.target_hours}h")

    t0 = time.time(); procesados = 0; fallos = 0
    for vid, segs in stream_videos():
        if hrs >= args.target_hours:
            log(f"TARGET alcanzado: {hrs:.2f}h >= {args.target_hours}h"); break
        if args.max_videos and procesados >= args.max_videos:
            log(f"MAX_VIDEOS alcanzado: {procesados}"); break
        dst = os.path.join(SALIDA, f"visper_{vid}")
        if f"visper_{vid}" in done:
            continue
        raw = os.path.join(args.raw_tmp, vid + ".mp4")
        if not descargar(vid, raw, args.sleep):
            fallos += 1
            if fallos % 10 == 0: log(f"  (descarga fallo #{fallos}, ultimo {vid})")
            continue
        try:
            ok, h = crop_video(raw, segs, vproc, dst)
        except Exception as e:
            log(f"  crop fallo {vid}: {e}"); ok, h = 0, 0.0
        finally:
            if os.path.exists(raw):
                os.remove(raw)  # liberar disco
        hrs += h; procesados += 1
        if ok == 0 and os.path.isdir(dst) and not os.listdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        rate = (time.time() - t0) / max(procesados, 1)
        falta = max(0.0, args.target_hours - hrs)
        h_por_video = hrs / max(procesados, 1)
        eta_min = (falta / max(h_por_video, 1e-6)) * rate / 60
        if procesados % 5 == 0 or procesados == 1:
            log(f"  {procesados} vids | {hrs:.2f}h/{args.target_hours}h | {ok} clips ult | "
                f"{rate:.0f}s/vid | ETA ~{eta_min:.0f}min")
        if procesados % args.backup_cada == 0:
            backup(args.bucket)

    backup(args.bucket)
    el = (time.time() - t0) / 60
    log(f"DONE | horas={hrs:.2f} | videos procesados={procesados} fallos={fallos} | {el:.0f}min")
    log(f"npz en {SALIDA}/visper_*")

if __name__ == "__main__":
    main()
