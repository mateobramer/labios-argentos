#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scorea el mini test set (grabado con build_testset.py) con ft05 o ViSpeR.
Imports perezosos segun --model, asi corre en el env que corresponda:

  ft05   (env mvsr)  : ~/miniforge3/envs/mvsr/bin/python  score_selftest.py --model ft05
  visper (env visper): ~/miniconda3/envs/visper/bin/python score_selftest.py --model visper
"""
import os, sys, csv, re, unicodedata, argparse

def norm(s):
    s=s.lower().strip().replace("ñ","\x00"); s=unicodedata.normalize("NFD",s)
    s="".join(c for c in s if unicodedata.category(c)!="Mn"); s=s.replace("\x00","ñ")
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9ñ ]"," ",s)).strip()
def ed(a,b):
    n,m=len(a),len(b)
    if n==0: return m
    if m==0: return n
    p=list(range(m+1))
    for i in range(1,n+1):
        c=[i]+[0]*m
        for j in range(1,m+1): c[j]=min(p[j]+1,c[j-1]+1,p[j-1]+(0 if a[i-1]==b[j-1] else 1))
        p=c
    return p[m]
def rate(pairs, lvl):
    te=tn=0
    for r,h in pairs:
        a,b=(r.split(),h.split()) if lvl=="w" else (r,h)
        te+=ed(a,b); tn+=len(a)
    return 100.0*te/max(tn,1)

def load_ft05():
    # MVSR_REPO pisa el default; el checkpoint sale del repo (o FT05_CKPT). Ver .env.example.
    REPO=os.path.expanduser(os.environ.get("MVSR_REPO", "~/Desktop/Visual_Speech_Recognition_for_Multiple_Languages"))
    sys.path.insert(0, REPO)
    import torch
    from pipelines.data.transforms import VideoTransform
    from pipelines.model import AVSR
    mc=REPO+"/benchmarks/CMUMOSEAS/models/es/CMUMOSEAS_V_ES_WER44.5/model.json"
    lm=REPO+"/benchmarks/CMUMOSEAS/language_models/es/lm_es/model.pth"
    lmc=REPO+"/benchmarks/CMUMOSEAS/language_models/es/lm_es/model.json"
    _root=os.environ.get("LABIOS_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck=os.environ.get("FT05_CKPT") or os.path.join(_root, "modelos/ft05_espnet1.pth")
    m=AVSR("video",ck,mc,lm,lmc,0.0,0.1,0.4,30,"cpu"); vt=VideoTransform(speed_rate=1)
    def infer(rois):
        return norm(m.infer(vt(torch.tensor(rois))).replace("<space>"," ").replace("▁"," ").replace("<eos>",""))
    return infer

def load_visper():
    REPO=os.path.expanduser(os.environ.get("VISPER_DIR", "~/Desktop/visper")); sys.path.insert(0, REPO)
    import torch
    from datamodule.transforms import VideoTransform
    from lightning_vsr import ModelModule
    from visper_zeroshot import build_cfg
    mm=ModelModule(build_cfg()); mm.eval(); mm.to("cpu")
    vt=VideoTransform(subset="test")
    def infer(rois):
        import torch
        v=torch.tensor(rois).unsqueeze(1).repeat(1,3,1,1).float()
        return norm(mm(vt(v), lang=4))
    return infer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ft05","visper"])
    ap.add_argument("--testdir", default=os.path.expanduser("~/vsr_selftest"))
    args=ap.parse_args()
    import numpy as np
    print(f"[score] cargando {args.model}...", flush=True)
    infer = load_ft05() if args.model=="ft05" else load_visper()
    rows=list(csv.DictReader(open(os.path.join(args.testdir,"manifest.csv"))))
    pairs=[]
    for r in rows:
        rois=np.load(os.path.join(args.testdir, r["clip"]+".npz"))["rois"]
        ref, hyp = norm(r["texto"]), infer(rois)
        pairs.append((ref,hyp))
        print(f"  REAL : {ref}\n  {args.model.upper():5}: {hyp}\n")
    print(f"===== {args.model} sobre {len(pairs)} clips propios =====")
    print(f"  WER = {rate(pairs,'w'):.2f}   CER = {rate(pairs,'c'):.2f}")

if __name__=="__main__":
    main()
