#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Arma los CSVs de train/val para la calibracion al hablante, desde ~/vsr_personal/<persona>/
(clips grabados con la pagina /calibrar de demo_web). Val = 1 de cada 8. Sin test: el test
es el uso en vivo (y experiments/10 ya valido el metodo con test congelado).

Uso:  python armar_splits_cal.py <persona> [--out <dir>]
"""
import os, csv, glob, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("persona")
    ap.add_argument("--src", default="", help="default: ~/vsr_personal/<persona>")
    ap.add_argument("--out", default="", help="default: mismo dir")
    args = ap.parse_args()
    src = os.path.expanduser(args.src or f"~/vsr_personal/{args.persona}")
    out = os.path.expanduser(args.out or src)

    clips = sorted(glob.glob(os.path.join(src, "clip_*.npz")))
    if len(clips) < 15:
        print(f"ERROR: solo {len(clips)} clips en {src} — grabá al menos ~20 (ideal 40).")
        raise SystemExit(1)
    rows = []
    for p in clips:
        stem = os.path.splitext(os.path.basename(p))[0]
        txt = os.path.join(src, stem + ".txt")
        if not os.path.exists(txt):
            print(f"  [warn] {stem} sin .txt, salteado"); continue
        rows.append({"clip": stem,
                     "texto": open(txt, encoding="utf-8").read().strip(),
                     "n_frames": int(np.load(p)["rois"].shape[0]),
                     "npz": f"cal_{args.persona}/{stem}.npz"})
    val = rows[7::8]                      # 1 de cada 8 a val
    train = [r for r in rows if r not in val]
    for split, rws in (("train", train), ("val", val)):
        path = os.path.join(out, f"cal_{split}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip", "texto", "n_frames", "npz"])
            w.writeheader(); w.writerows(rws)
        print(f"  {path}: {len(rws)} clips")
    print(f"[splits] {args.persona}: train {len(train)} / val {len(val)}")

if __name__ == "__main__":
    main()
