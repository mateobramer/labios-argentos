#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter ViSpeR -> clips (formato que espera preprocessing/preprocesar.py).

Lee un JSON de ViSpeR (keyed por video-id de YouTube; cada valor = lista de segmentos
con start/end/label/landmarks), baja cada video con yt-dlp y corta cada segmento a
  data/clips/visper_<id>/clip_NNNN.mp4  (+ .txt con la transcripcion `label`)
que luego procesa `preprocesar.py` -> data/processed/lip_rois/visper_<id>/clip_NNNN.npz

Uso:
  python curriculum/visper_a_clips.py --json <chunk.json> --limit-videos 2 --max-clips 6
"""
import os, sys, json, subprocess, argparse, random, shutil, time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.path.join(RAIZ, "data", "clips")

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode

def descargar(vid, dst):
    url = f"https://www.youtube.com/watch?v={vid}"
    # 360p mp4 (fmt 18) o fallback <=480; sin audio hace falta -> igual bajamos con audio y lo ignoramos
    return run(["yt-dlp", "-q", "--no-warnings",
                "-f", "18/bv*[height<=480]+ba/b[height<=480]",
                "-o", dst, url])

def cortar(video, start, end, out):
    dur = max(0.0, float(end) - float(start))
    # decode desde el inicio para corte exacto; sin audio; mantiene fps original
    return run(["ffmpeg", "-y", "-i", video, "-ss", f"{start}", "-t", f"{dur}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", out])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--max-clips", type=int, default=0, help="tope de clips por video (0=todos)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tmp", default=os.path.join(RAIZ, "data", "_visper_tmp"))
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))
    vids = list(data.keys())
    random.Random(args.seed).shuffle(vids)
    if args.limit_videos:
        vids = vids[:args.limit_videos]
    os.makedirs(args.tmp, exist_ok=True)

    tot_clips = tot_seg_s = 0; vid_ok = vid_fail = 0
    t0 = time.time()
    for i, vid in enumerate(vids, 1):
        segs = data[vid]
        if args.max_clips:
            segs = segs[:args.max_clips]
        raw = os.path.join(args.tmp, f"{vid}.mp4")
        if not os.path.exists(raw):
            if descargar(vid, raw) != 0 or not os.path.exists(raw):
                print(f"  [{i}/{len(vids)}] {vid}: descarga FALLO", flush=True); vid_fail += 1; continue
        titulo = f"visper_{vid}"
        dst = os.path.join(CLIPS_DIR, titulo); os.makedirs(dst, exist_ok=True)
        n = 0
        for s in segs:
            out = os.path.join(dst, f"clip_{n+1:04d}.mp4")
            if cortar(raw, s["start"], s["end"], out) == 0 and os.path.exists(out):
                with open(out[:-4] + ".txt", "w", encoding="utf-8") as f:
                    f.write(str(s.get("label", "")).strip() + "\n")
                n += 1; tot_seg_s += float(s["end"]) - float(s["start"])
        tot_clips += n; vid_ok += 1
        print(f"  [{i}/{len(vids)}] {vid}: {n} clips ({time.time()-t0:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"\n===== ViSpeR->clips =====")
    print(f"videos OK={vid_ok} FAIL={vid_fail} | clips={tot_clips} | horas_clip={tot_seg_s/3600:.2f}h | {el:.0f}s")
    print(f"tmp crudos en {args.tmp} (borrar tras preprocesar); clips en {CLIPS_DIR}/visper_*")

if __name__ == "__main__":
    main()
