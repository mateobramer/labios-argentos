#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura ΔWER-vs-CER de la grilla (env ptt, matplotlib).
Lee data/grid_puntos.json y genera umbral_cer_llm.png (en este mismo dir)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
puntos = json.load(open(DATA + "/grid_puntos.json", encoding="utf-8"))

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
ax.axhline(0, color="#888", lw=0.8, zorder=1)

MARCA = {"658": ("o", "test-658 (YouTube)"), "Rioplatense": ("o", "test-658 (YouTube)"),
         "fede": ("s", "self-test (limpio)"), "amigo": ("s", "self-test (limpio)"),
         "Selftest": ("s", "self-test (limpio)")}
vistos = set()
for p in puntos:
    tag, test = p["celda"].split("/")
    if tag in ("ft05", "ft07") and test in ("fede", "amigo"):
        continue  # variantes con-LM: van aparte en la tabla, no en la curva principal
    m, lbl = MARCA.get(test, ("o", test))
    sig = p["ic_resc"][0] > 0
    color = "#1c7ed6" if m == "s" else "#e8590c"
    lbl_full = lbl if lbl not in vistos else None
    vistos.add(lbl)
    ax.errorbar(p["cer_base"], p["delta_resc"],
                yerr=[[p["delta_resc"] - p["ic_resc"][0]], [p["ic_resc"][1] - p["delta_resc"]]],
                fmt=m, color=color, markersize=9 if sig else 7,
                markerfacecolor=color if sig else "white",
                capsize=3, lw=1.2, label=lbl_full, zorder=3)
    nombre = tag.replace("visper-", "")
    ax.annotate(nombre, (p["cer_base"], p["delta_resc"]),
                textcoords="offset points", xytext=(7, 6), fontsize=7.5, color="#444")

ax.annotate("en YouTube: nulo a todo CER\n(IC angostos, n=658)",
            xy=(34, 0.35), fontsize=8.5, color="#e8590c", ha="center",
            xytext=(34, -1.1), arrowprops=dict(arrowstyle="-", color="#e8590c", lw=0.7))
ax.annotate("en limpio: crece al bajar el CER\n(sig. a 14.6 y 31.9)",
            xy=(16.5, 3.0), fontsize=8.5, color="#1c7ed6", ha="center",
            xytext=(22, 4.9), arrowprops=dict(arrowstyle="-", color="#1c7ed6", lw=0.7))

ax.set_xlabel("CER base del sistema (%)")
ax.set_ylabel("Δ WER por n-best rescoring (+ = mejora)")
ax.set_title("¿A qué CER empieza a servir el LLM? — qwen3:4b top-5, IC95 pareado\n"
             "relleno = significativo · vacío = no significativo", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.25)
out = AQUI + "/umbral_cer_llm.png"
fig.tight_layout()
fig.savefig(out)
print("figura ->", out)
