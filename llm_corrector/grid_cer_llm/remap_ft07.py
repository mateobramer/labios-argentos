#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remapea ft07_best.pth (espnet2, claves de Gimeno) -> espnet1 (claves de mpc001/AVSR).
Mismo procedimiento que ft05_espnet1 (docs/experiments/06): el mapa se deriva POR POSICION
comparando es_remapped.pth (espnet2) vs el model.pth de CMU-MOSEAS (espnet1) — son los
mismos 767 tensores en el mismo orden (rename puro). Se AUTOVALIDA re-derivando
ft05_espnet1 y comparando con el archivo existente. Env mvsr."""
import torch, os

MOD = os.path.expanduser("~/Desktop/labios-argentos/modelos")
CMU = os.path.expanduser("~/Desktop/Visual_Speech_Recognition_for_Multiple_Languages"
                         "/benchmarks/CMUMOSEAS/models/es/CMUMOSEAS_V_ES_WER44.5/model.pth")

es2 = torch.load(f"{MOD}/es_remapped.pth", map_location="cpu")
es2 = es2.get("model", es2) if isinstance(es2, dict) and "model" in es2 else es2
e1 = torch.load(CMU, map_location="cpu")
e1 = e1.get("model", e1) if isinstance(e1, dict) and "model" in e1 else e1

k2, k1 = list(es2.keys()), list(e1.keys())
assert len(k2) == len(k1), f"cantidad de tensores distinta: {len(k2)} vs {len(k1)}"
iguales = sum(torch.equal(es2[a], e1[b]) for a, b in zip(k2, k1))
print(f"[remap] {len(k2)} tensores; identicos por valor en orden: {iguales}")
assert iguales > len(k2) * 0.95, "el orden posicional no coincide — abortar"
mapa = dict(zip(k2, k1))

def remap(src, dst):
    sd = torch.load(src, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    faltan = [k for k in sd if k not in mapa]
    assert not faltan, f"claves fuera del mapa: {faltan[:5]}"
    out = {mapa[k]: v for k, v in sd.items()}
    torch.save(out, dst)
    print(f"[remap] {os.path.basename(src)} -> {os.path.basename(dst)} ({len(out)} tensores)")
    return out

# autovalidacion: re-derivar ft05_espnet1 y comparar con el existente
val = remap(f"{MOD}/ft05_best.pth", "/tmp/ft05_espnet1_rederivado.pth")
ref = torch.load(f"{MOD}/ft05_espnet1.pth", map_location="cpu")
ok = all(torch.equal(val[k], ref[k]) for k in ref)
print(f"[remap] VALIDACION ft05: {'IDENTICO al existente ✓' if ok else 'DIFIERE ✗ — revisar'}")
assert ok

remap(f"{MOD}/ft07_best.pth", f"{MOD}/ft07_espnet1.pth")
print("[remap] LISTO")
