#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Zero-shot del modelo VSR multilingue de ViSpeR (visper_vsr_base.pth, language token <es>=4)
sobre nuestro test-658 rioplatense. Alimenta nuestros npz (T,96,96 gris, mean-face 96x96)
directo al VideoTransform(test) + encoder + beam search. Metricas identicas a zeroshot.py de mpc001.

Uso (env conda visper):
  python visper_zeroshot.py --test-csv <test.csv> [--limit N] [--out-prefix runs/visper_es]
"""
import os, sys, csv, re, time, argparse, unicodedata, random
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import numpy as np, torch
from omegaconf import OmegaConf
from datamodule.transforms import VideoTransform
from lightning_vsr import ModelModule

# ---------- metricas (identicas a mpc001 zeroshot.py) ----------
def norm(s):
    s = s.lower().strip().replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def edits(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0]*m; ai = a[i-1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1] + (0 if ai == b[j-1] else 1))
        prev = cur
    return prev[m]

def corpus_rate(pairs, level):
    te = tn = 0; per = []
    for ref, hyp in pairs:
        r, h = (ref.split(), hyp.split()) if level == "word" else (ref, hyp)
        e = edits(r, h); n = len(r); te += e; tn += n; per.append((e, n))
    return 100.0*te/max(tn, 1), per

def bootstrap_ci(per, iters=2000, seed=1234):
    rnd = random.Random(seed); N = len(per); vals = []
    for _ in range(iters):
        te = tn = 0
        for _ in range(N):
            e, n = per[rnd.randrange(N)]; te += e; tn += n
        vals.append(100.0*te/max(tn, 1))
    vals.sort()
    return (vals[int(0.975*iters)] - vals[int(0.025*iters)]) / 2.0

def build_cfg():
    vb = OmegaConf.load(os.path.join(REPO, "conf/model/visual_backbone/resnet_conformer.yaml"))
    cfg = OmegaConf.create({
        "data": {"modality": "video"},
        "model": {
            "spm_dict": "spm/unigram/unigram_units.txt",
            "spm_model": "spm/unigram/unigram.model",
            "visual_backbone": vb,
        },
        "ckpt_path": os.path.join(REPO, "visper_vsr_base.pth"),
        "transfer_frontend": False,
        "transfer_encoder": False,
    })
    OmegaConf.resolve(cfg)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-prefix", default="runs/visper_es")
    ap.add_argument("--lang", default="spanish")
    ap.add_argument("--data-root", default="", help="prefijo para paths npz relativos")
    ap.add_argument("--load-ckpt", default="", help="state_dict fine-tuneado a evaluar (best.pth); si vacío usa el base")
    args = ap.parse_args()

    LANG_TOKEN = {'english': 2, 'arabic': 3, 'spanish': 4, 'french': 5, 'chinese': 6}
    lang = LANG_TOKEN[args.lang]

    print("[load] construyendo ModelModule (ViSpeR)...", flush=True)
    mm = ModelModule(build_cfg())
    if args.load_ckpt:
        sd = torch.load(args.load_ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        mm.model.load_state_dict(sd)
        print(f"[load] pesos fine-tuneados: {args.load_ckpt}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mm.eval(); mm.to(dev)
    print(f"[load] device={dev}", flush=True)
    vt = VideoTransform(subset="test")

    rows = list(csv.DictReader(open(args.test_csv)))
    if args.limit: rows = rows[:args.limit]
    print(f"[run] {len(rows)} clips, lang={args.lang}({lang})", flush=True)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    pairs = []; inf_lines = []; t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            npzp = r["npz"] if os.path.isabs(r["npz"]) else os.path.join(args.data_root, r["npz"])
            rois = np.load(npzp)["rois"]                 # (T,96,96) uint8
            v = torch.tensor(rois).unsqueeze(1).repeat(1, 3, 1, 1).float()  # (T,3,96,96)
            data = vt(v)                                      # (T,1,88,88)
            with torch.no_grad():
                hyp_raw = mm(data, lang=lang)
        except Exception as e:
            hyp_raw = ""; print(f"  [warn] clip {i} fallo: {e}", flush=True)
        ref, hyp = norm(r["texto"]), norm(hyp_raw)
        pairs.append((ref, hyp)); inf_lines.append(f"{ref}#{hyp}")
        if i % 10 == 0 or i == len(rows):
            el = time.time()-t0; w, _ = corpus_rate(pairs, "word")
            print(f"  {i}/{len(rows)} WER_parcial={w:.2f} ({el/i:.1f}s/clip, ETA {el/i*(len(rows)-i)/60:.0f}min)", flush=True)

    wer, pw = corpus_rate(pairs, "word"); cer, pc = corpus_rate(pairs, "char")
    res = f"%WER: {wer:.4f} ± {bootstrap_ci(pw):.4f}\n%CER: {cer:.4f} ± {bootstrap_ci(pc):.4f}\n"
    open(args.out_prefix + "_test.inf", "w").write("\n".join(inf_lines) + "\n")
    open(args.out_prefix + "_test.wer", "w").write(res)
    print("\n===== ViSpeR zero-shot (test-658) =====\n" + res)
    print("ejemplos (ref#hyp):")
    for ln in inf_lines[:8]: print("  ", ln)

if __name__ == "__main__":
    main()
