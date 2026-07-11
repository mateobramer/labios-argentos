#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diversidad de candidatas por celda: ¿el LM externo achata el beam?
Metricas por celda: candidatas unicas promedio (de 5), y distancia media
entre la 1-best y las otras candidatas (palabras editadas / largo)."""
import json, glob, os

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

def textos(d):
    return [c[0] if isinstance(c, list) else c for c in d["cands"] if (c[0] if isinstance(c, list) else c)][:5]

ARCHIVOS = {
    "ft05/658 GPU (sin LM)": "cands_Rioplatense_ft05.json",
    "ft07/658 GPU (sin LM)": "cands_Rioplatense_ft07.json",
    "ft05/selftest GPU (sin LM)": "cands_Selftest_ft05.json",
    "ft07/selftest GPU (sin LM)": "cands_Selftest_ft07.json",
    "ft05/fede local (CON LM)": "cands_fede_ft05.json",
    "ft05/amigo local (CON LM)": "cands_amigo_ft05.json",
    "ft07/amigo local (CON LM)": "cands_amigo_ft07.json",
    "visper/658 (sin LM)": "cands_658_fp32.json",
    "visper/fede (sin LM)": "cands100_scored.json",
    "visper/amigo (sin LM)": "cands_amigo.json",
}
print(f"{'celda':30} {'unicas/5':>9} {'dist media 1best↔resto':>24}")
for nombre, f in ARCHIVOS.items():
    p = os.path.join(DATA, f)
    if not os.path.exists(p):
        continue
    data = json.load(open(p, encoding="utf-8"))
    uni, dist, nn = 0.0, 0.0, 0
    for d in data:
        cs = textos(d)
        if len(cs) < 2:
            continue
        uni += len(set(cs))
        w0 = cs[0].split()
        dist += sum(ed(w0, c.split()) / max(len(w0), 1) for c in cs[1:]) / (len(cs) - 1)
        nn += 1
    print(f"{nombre:30} {uni/max(nn,1):9.2f} {dist/max(nn,1):24.3f}")
