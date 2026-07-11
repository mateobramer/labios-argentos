#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Efecto del rescoring por estrato de CER-por-clip DENTRO de cada celda grande.
Mismas frases, mismo modelo: la variacion de CER es natural por clip."""
import json, os
AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")

def ed(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0] * m
        for j in range(1, m + 1):
            c[j] = min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        p = c
    return p[m]

def wer_pairs(pairs):
    te = tn = 0
    for r, h in pairs:
        te += ed(r.split(), h.split()); tn += len(r.split())
    return 100.0 * te / max(tn, 1)

def textos(d):
    return [c[0] if isinstance(c, list) else c for c in d["cands"] if (c[0] if isinstance(c, list) else c)][:5]

CELDAS = {"visper-fp32/658": "cands_658_fp32.json", "ft05/658": "cands_Rioplatense_ft05.json",
          "visper-fp32/fede+amigo": None}
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 999)]

def analizar(nombre, data):
    print(f"\n== {nombre} (n={len(data)}) — delta WER del rescoring por estrato de CER del clip ==")
    for lo, hi in BUCKETS:
        grupo = []
        for d in data:
            if "resc" not in d: continue
            r, h = d["ref"], textos(d)[0]
            cer = 100.0 * ed(r, h) / max(len(r), 1)
            if lo <= cer < hi:
                grupo.append(d)
        if len(grupo) < 8:
            print(f"  CER {lo:>2}-{hi if hi<999 else '+':<3}: n={len(grupo):3} (muy chico)"); continue
        w1 = wer_pairs([(d["ref"], textos(d)[0]) for d in grupo])
        wr = wer_pairs([(d["ref"], d["resc"]) for d in grupo])
        print(f"  CER {lo:>2}-{hi if hi<999 else '+':<3}: n={len(grupo):3}  1best {w1:6.2f} -> resc {wr:6.2f}  (delta {w1-wr:+.2f})")

analizar("visper-fp32/658", json.load(open(DATA + "/cands_658_fp32.json")))
analizar("ft05/658 (GPU)", json.load(open(DATA + "/cands_Rioplatense_ft05.json")))
pool = json.load(open(DATA + "/cands100_scored.json")) + json.load(open(DATA + "/cands_amigo.json"))
analizar("visper-fp32/selftest-150", pool)
