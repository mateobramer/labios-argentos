#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genera candidatas top-5 de ViSpeR para una celda de la matriz CER/LLM.
Env visper. Persistencia incremental + resumible (saltea clips ya hechos).

  ~/miniconda3/envs/visper/bin/python infer_cell.py --test 658 --quant fp32
  ~/miniconda3/envs/visper/bin/python infer_cell.py --test amigo --quant int8

--test 658   -> vsr_models/splits/test.csv (npz relativos al repo)
--test fede  -> ~/vsr_selftest/manifest.csv
--test amigo -> ~/Desktop/grabaciones/manifest.csv
fp32: encoder en MPS + beam40 CPU (config F2/F3). int8: todo CPU qnnpack (config 09 §G).
"""
import os, sys, csv, re, json, argparse, time, unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True, choices=["658", "fede", "amigo"])
ap.add_argument("--quant", default="fp32", choices=["fp32", "int8"])
args = ap.parse_args()

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
REPO = os.path.expanduser("~/Desktop/labios-argentos")
OUT = os.path.join(DATA, f"cands_{args.test}_{args.quant}.json")
NBEST, BEAM, LANG = 5, 40, 4

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.expanduser("~/Desktop/visper"))
import numpy as np, torch
from datamodule.transforms import VideoTransform
from lightning_vsr import ModelModule, get_beam_search_decoder
from visper_zeroshot import build_cfg
from espnet.asr.asr_utils import add_results_to_json as _arj

def norm(s):
    s = s.lower().strip().replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def filas():
    if args.test == "658":
        rows = list(csv.DictReader(open(os.path.join(REPO, "vsr_models/splits/test.csv"), encoding="utf-8")))
        return [(f"{r['titulo']}/{r['clip']}", os.path.join(REPO, r["npz"]), r["texto"]) for r in rows]
    d = os.path.expanduser("~/vsr_selftest" if args.test == "fede" else "~/Desktop/grabaciones")
    rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"), encoding="utf-8")))
    return [(r["clip"], os.path.join(d, r["clip"] + ".npz"), r["texto"]) for r in rows]

print(f"[cell {args.test}/{args.quant}] cargando ViSpeR...", flush=True)
mm = ModelModule(build_cfg())
mm.eval(); mm.to("cpu")
if args.quant == "int8":
    torch.backends.quantized.engine = "qnnpack"
    mm.model = torch.ao.quantization.quantize_dynamic(mm.model, {torch.nn.Linear}, dtype=torch.qint8)
    enc_dev = "cpu"   # el modelo cuantizado no puede ir a MPS
else:
    enc_dev = "mps" if torch.backends.mps.is_available() else "cpu"
mm.beam_search = get_beam_search_decoder(mm.model, mm.token_list, ctc_weight=0.1, beam_size=BEAM)
if enc_dev != "cpu":
    mm.model.encoder.to(enc_dev)
vt = VideoTransform(subset="test")

todo = filas()
hechos = {}
if os.path.exists(OUT):
    hechos = {d["clip"]: d for d in json.load(open(OUT, encoding="utf-8"))}
    print(f"[cell] resume: {len(hechos)}/{len(todo)} ya hechos", flush=True)
out = [hechos[c] for c, _, _ in todo if c in hechos]

t0 = time.time()
for k, (clip, npzp, texto) in enumerate(todo):
    if clip in hechos:
        continue
    try:
        rois = np.load(npzp)["rois"]
        v = torch.tensor(rois).unsqueeze(1).repeat(1, 3, 1, 1).float()
        data = vt(v)
        with torch.no_grad():
            enc, _ = mm.model.encoder(data.unsqueeze(0).to(enc_dev), None)
            enc = enc.squeeze(0).to("cpu")
            hyps = mm.beam_search(enc, lang=LANG)
        cands = [norm(_arj([h.asdict()], mm.token_list, LANG).replace("▁", " ").replace("<eos>", ""))
                 for h in hyps[:min(len(hyps), NBEST)]]
        out.append({"clip": clip, "ref": norm(texto), "cands": cands})
    except Exception as e:
        out.append({"clip": clip, "ref": norm(texto), "cands": [""], "error": str(e)})
        print(f"[cell] ERROR {clip}: {e}", flush=True)
    if len(out) % 10 == 0 or k == len(todo) - 1:
        json.dump(out, open(OUT, "w"), ensure_ascii=False)
        el = time.time() - t0
        print(f"[cell {args.test}/{args.quant}] {len(out)}/{len(todo)} ({el/60:.0f} min, "
              f"{el/max(len(out)-len(hechos),1):.1f} s/clip)", flush=True)
json.dump(out, open(OUT, "w"), ensure_ascii=False)
print(f"[cell {args.test}/{args.quant}] LISTO -> {OUT}", flush=True)
