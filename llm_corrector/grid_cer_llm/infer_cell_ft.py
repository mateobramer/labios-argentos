#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Candidatas top-5 de los modelos 50M (ft05/ft07 remapeados a espnet1) para la grilla CER/LLM.
Env mvsr:  ~/miniforge3/envs/mvsr/bin/python infer_cell_ft.py --model ft05 --test 658
Mismo decode que las evals del 50M local: beam 30, ctc 0.1, LM externo 0.4 (config
CMUMOSEAS_V_ES_WER44.5, validado en docs/experiments/06). Persistencia incremental + resumible."""
import os, sys, csv, re, json, argparse, time, unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["ft05", "ft07"])
ap.add_argument("--test", required=True, choices=["658", "fede", "amigo"])
args = ap.parse_args()

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
LREPO = os.path.expanduser("~/Desktop/labios-argentos")
REPO = os.path.expanduser("~/Desktop/Visual_Speech_Recognition_for_Multiple_Languages")
sys.path.insert(0, REPO)
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, torch
from pipelines.data.transforms import VideoTransform
from pipelines.model import AVSR
from espnet.asr.asr_utils import add_results_to_json

OUT = os.path.join(DATA, f"cands_{args.test}_{args.model}.json")
NBEST = 5

def norm(s):
    s = s.lower().strip().replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def clean(s):
    return s.replace("<space>", " ").replace("▁", " ").replace("<eos>", "")

def filas():
    if args.test == "658":
        rows = list(csv.DictReader(open(os.path.join(LREPO, "vsr_models/splits/test.csv"), encoding="utf-8")))
        return [(f"{r['titulo']}/{r['clip']}", os.path.join(LREPO, r["npz"]), r["texto"]) for r in rows]
    d = os.path.expanduser("~/vsr_selftest" if args.test == "fede" else "~/Desktop/grabaciones")
    rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"), encoding="utf-8")))
    return [(r["clip"], os.path.join(d, r["clip"] + ".npz"), r["texto"]) for r in rows]

mc = REPO + "/benchmarks/CMUMOSEAS/models/es/CMUMOSEAS_V_ES_WER44.5/model.json"
lm = REPO + "/benchmarks/CMUMOSEAS/language_models/es/lm_es/model.pth"
lmc = REPO + "/benchmarks/CMUMOSEAS/language_models/es/lm_es/model.json"
ck = os.path.join(LREPO, "modelos", f"{args.model}_espnet1.pth")
print(f"[cell {args.test}/{args.model}] cargando AVSR ({os.path.basename(ck)})...", flush=True)
model = AVSR("video", ck, mc, lm, lmc, 0.0, 0.1, 0.4, 30, "cpu")
vt = VideoTransform(speed_rate=1)

todo = filas()
hechos = {}
if os.path.exists(OUT):
    hechos = {d["clip"]: d for d in json.load(open(OUT, encoding="utf-8"))}
    print(f"[cell] resume: {len(hechos)}/{len(todo)} ya hechos", flush=True)
out = [hechos[c] for c, _, _ in todo if c in hechos]

t0, nuevos = time.time(), 0
for clip, npzp, texto in todo:
    if clip in hechos:
        continue
    try:
        data = vt(torch.tensor(np.load(npzp)["rois"]))
        with torch.no_grad():
            enc = model.model.encode(data.to("cpu"))
            hyps = model.beam_search(enc)
        cands = [norm(clean(add_results_to_json([h.asdict()], model.token_list)))
                 for h in hyps[:min(len(hyps), NBEST)]]
        out.append({"clip": clip, "ref": norm(texto), "cands": cands})
    except Exception as e:
        out.append({"clip": clip, "ref": norm(texto), "cands": [""], "error": str(e)})
        print(f"[cell] ERROR {clip}: {e}", flush=True)
    nuevos += 1
    if len(out) % 10 == 0:
        json.dump(out, open(OUT, "w"), ensure_ascii=False)
        el = time.time() - t0
        print(f"[cell {args.test}/{args.model}] {len(out)}/{len(todo)} ({el/60:.0f} min, {el/max(nuevos,1):.1f} s/clip)", flush=True)
json.dump(out, open(OUT, "w"), ensure_ascii=False)
print(f"[cell {args.test}/{args.model}] LISTO -> {OUT}", flush=True)
