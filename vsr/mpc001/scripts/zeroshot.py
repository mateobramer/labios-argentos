#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Zero-shot del modelo VSR de mpc001 (CMU-MOSEAS ES, 44.5 WER) sobre nuestro test-658
rioplatense. Alimenta nuestros npz (T,96,96 grises, ya croppeados con el MISMO
video_process + 20words_mean_face que espera el modelo) directo al VideoTransform + AVSR.infer.

Uso:
  python zeroshot.py --config <ini> --test-csv <csv> --repo <dir> \
      [--limit N] [--device cpu] [--no-lm] [--out-prefix runs/es_zeroshot]
"""
import os, sys, csv, re, json, time, argparse, unicodedata, random

import numpy as np
import torch

# --------- limpieza de tokens especiales del modelo char (usa '<space>' literal) ---------
def clean_hyp(s):
    s = s.replace("<space>", " ")
    for tok in ("<eos>", "<unk>", "<blank>", "<sos>"):
        s = s.replace(tok, " ")
    return s

# --------- normalizacion identica a la de las refs (lower, sin acentos, ñ ok, sin punt) ---------
def norm(s):
    s = s.lower().strip()
    s = s.replace("ñ", "\x00")                       # proteger la ñ
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # sacar acentos
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def edits(a, b):
    # distancia de edicion (Levenshtein) sobre listas/strings
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]

def corpus_rate(pairs, level):
    # pairs: lista de (ref_norm, hyp_norm). level: 'word' | 'char'
    tot_e = tot_n = 0
    per = []
    for ref, hyp in pairs:
        if level == "word":
            r, h = ref.split(), hyp.split()
        else:
            r, h = ref, hyp
        e = edits(r, h); n = len(r)
        tot_e += e; tot_n += n
        per.append((e, n))
    rate = 100.0 * tot_e / max(tot_n, 1)
    return rate, per

def bootstrap_ci(per, iters=2000, seed=1234):
    rnd = random.Random(seed)
    N = len(per)
    vals = []
    for _ in range(iters):
        te = tn = 0
        for _ in range(N):
            e, n = per[rnd.randrange(N)]
            te += e; tn += n
        vals.append(100.0 * te / max(tn, 1))
    vals.sort()
    lo = vals[int(0.025 * iters)]
    hi = vals[int(0.975 * iters)]
    return (hi - lo) / 2.0  # semi-ancho del IC95%

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--repo", required=True, help="dir del repo mpc001 (para sys.path)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--out-prefix", default="runs/es_zeroshot")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from configparser import ConfigParser
    from pipelines.data.transforms import VideoTransform
    from pipelines.model import AVSR

    cfg = ConfigParser(); cfg.read(args.config)
    base = args.repo
    def P(p): return p if os.path.isabs(p) else os.path.join(base, p)
    model_path = P(cfg.get("model", "model_path"))
    model_conf = P(cfg.get("model", "model_conf"))
    rnnlm      = None if args.no_lm else P(cfg.get("model", "rnnlm"))
    rnnlm_conf = None if args.no_lm else P(cfg.get("model", "rnnlm_conf"))
    penalty    = cfg.getfloat("decode", "penalty")
    ctc_weight = cfg.getfloat("decode", "ctc_weight")
    lm_weight  = 0.0 if args.no_lm else cfg.getfloat("decode", "lm_weight")
    beam_size  = cfg.getint("decode", "beam_size")

    print(f"[cfg] model={model_path}\n[cfg] lm={rnnlm} (lm_weight={lm_weight}) beam={beam_size} ctc_w={ctc_weight}")
    for f in [model_path, model_conf] + ([rnnlm, rnnlm_conf] if rnnlm else []):
        assert os.path.isfile(f), f"falta archivo: {f}"

    device = torch.device(args.device)
    print("[load] construyendo AVSR ...")
    model = AVSR("video", model_path, model_conf, rnnlm, rnnlm_conf,
                 penalty, ctc_weight, lm_weight, beam_size, device)
    # nuestros npz ya estan a 25fps (== model v_fps) => speed_rate=1, sin resample
    vt = VideoTransform(speed_rate=1)

    rows = list(csv.DictReader(open(args.test_csv)))
    if args.limit: rows = rows[:args.limit]
    print(f"[run] {len(rows)} clips de test en {args.device}")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    pairs = []
    t0 = time.time()
    inf_lines = []
    for i, r in enumerate(rows, 1):
        rois = np.load(r["npz"])["rois"]          # (T,96,96) uint8
        v = torch.tensor(rois)                     # (T,96,96)
        data = vt(v)                               # (1,T,88,88) float normalizado
        try:
            hyp_raw = model.infer(data)
        except Exception as e:
            hyp_raw = ""
            print(f"  [warn] clip {i} fallo: {e}")
        ref, hyp = norm(r["texto"]), norm(hyp_raw)
        pairs.append((ref, hyp))
        inf_lines.append(f"{ref}#{hyp}")
        if i % 20 == 0 or i == len(rows):
            el = time.time() - t0
            w, _ = corpus_rate(pairs, "word")
            print(f"  {i}/{len(rows)}  WER_parcial={w:.2f}  ({el/i:.2f}s/clip, ETA {el/i*(len(rows)-i)/60:.1f}min)", flush=True)

    wer, per_w = corpus_rate(pairs, "word")
    cer, per_c = corpus_rate(pairs, "char")
    wer_ci = bootstrap_ci(per_w)
    cer_ci = bootstrap_ci(per_c)

    with open(args.out_prefix + "_test.inf", "w") as f:
        f.write("\n".join(inf_lines) + "\n")
    res = f"%WER: {wer:.6f} ± {wer_ci:.6f}\n%CER: {cer:.6f} ± {cer_ci:.6f}\n"
    with open(args.out_prefix + "_test.wer", "w") as f:
        f.write(res)
    print("\n=========== RESULTADO ZERO-SHOT (test-658) ===========")
    print(res)
    print("ejemplos (ref#hyp):")
    for ln in inf_lines[:8]:
        print("  ", ln)

if __name__ == "__main__":
    main()
