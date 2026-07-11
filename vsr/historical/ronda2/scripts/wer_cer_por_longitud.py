#!/usr/bin/env python3
"""WER y CER por longitud de frase (en N de palabras de la REFERENCIA), desde un .inf (ref#hyp).

Misma convencion que el helper de la casa (.claude/skills/resultados/metricas.py):
  - WER: edit-distance a nivel PALABRA, micro-average (suma S+D+I / suma palabras de ref).
  - CER: edit-distance a nivel CARACTER (incluye espacios), micro-average.
Agrupa por cantidad de palabras de la referencia y reporta, por bucket:
  palabras, n_frases, wer, cer, palabras_ref (total de palabras de ref en el bucket).

Uso:
  python wer_cer_por_longitud.py <archivo.inf> [--csv salida.csv] [--json salida.json]
"""
import argparse
import csv
import json
import sys


def editdist(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def analizar(path):
    B = {}
    total = 0
    with open(path, encoding="utf-8") as f:
        for linea in f:
            linea = linea.rstrip("\n")
            if "#" not in linea:
                continue
            ref, hyp = linea.split("#", 1)
            ref, hyp = ref.strip(), hyp.strip()
            rw, hw = ref.split(), hyp.split()
            nw = len(rw)
            if nw == 0:
                continue
            total += 1
            ew, tw = editdist(rw, hw), len(rw)
            ec, tc = editdist(list(ref), list(hyp)), len(ref)
            b = B.setdefault(nw, {"n": 0, "ew": 0, "tw": 0, "ec": 0, "tc": 0})
            b["n"] += 1
            b["ew"] += ew
            b["tw"] += tw
            b["ec"] += ec
            b["tc"] += tc
    filas = []
    for nw in sorted(B):
        b = B[nw]
        wer = 100.0 * b["ew"] / b["tw"] if b["tw"] else 0.0
        cer = 100.0 * b["ec"] / b["tc"] if b["tc"] else 0.0
        filas.append({
            "palabras": nw,
            "n_frases": b["n"],
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            "palabras_ref": b["tw"],
        })
    return filas, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inf")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    a = ap.parse_args()
    filas, total = analizar(a.inf)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["palabras", "n_frases", "wer", "cer", "palabras_ref"])
            w.writeheader()
            w.writerows(filas)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(filas, f, ensure_ascii=False, indent=2)
    # tabla a stdout
    print(f"# {a.inf}  (total frases={total})")
    print(f"{'palabras':>8} {'n_frases':>8} {'wer':>7} {'cer':>7} {'pal_ref':>8}")
    for r in filas:
        print(f"{r['palabras']:>8} {r['n_frases']:>8} {r['wer']:>7.2f} {r['cer']:>7.2f} {r['palabras_ref']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
