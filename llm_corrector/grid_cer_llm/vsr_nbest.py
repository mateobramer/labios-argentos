#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variante de vsr_main.py que dumpea las TOP-5 candidatas del beam por clip (JSON).
Corre DENTRO de ~/evaluating-end2end-spanish-lipreading (env vsr-factors, GPU).
Mismo decode que las evals oficiales de los ft (beam 30, ctc 0.1, sin LM externo).

  python vsr_nbest.py --database Rioplatense --scenario zero-shot \
      --load-vsr ~/ckpts/ft05_best.pth --out ~/out/cands_658_ft05.json
"""
from src.tasks.asr import ASRTask
from src.bin.asr_inference import Speech2Text

import os, sys, csv, json, yaml, argparse, re, unicodedata
from pathlib import Path
import torch
from tqdm import tqdm
from src.utils import *
from src.Transforms import *

NBEST = 5

def norm(s):
    s = s.lower().strip().replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", default=0, type=int)
    p.add_argument("--database", required=True, type=str)
    p.add_argument("--scenario", default="zero-shot", type=str)
    p.add_argument("--vsr-config-file", default="./configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml", type=str)
    p.add_argument("--load-vsr", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    args = p.parse_args()

    with Path(args.vsr_config_file).open("r", encoding="utf-8") as f:
        vsr_config = argparse.Namespace(**yaml.safe_load(f))
    tokenizer, converter = get_tokenizer_converter(vsr_config.token_type, vsr_config.bpemodel, vsr_config.token_list)

    # normalizacion rioplatense SIEMPRE (mismos mean/std que la rama Rioplatense de vsr_main)
    tfm = Compose([Normalise(0.0, 250.0), Normalise(0.491, 0.166), CenterCrop((88, 88))])

    inf_conf = dict(vsr_config.inference_conf)
    inf_conf["nbest"] = NBEST
    speech2text = Speech2Text(
        asr_train_config=args.vsr_config_file,
        asr_model_file=args.load_vsr,
        lm_train_config=None, lm_file=None,   # sin LM: igual que las evals oficiales ft03-ft07
        **inf_conf,
    )
    loader = get_dataloader(args, vsr_config, dataset="test", transforms=tfm,
                            tokenizer=tokenizer, converter=converter)
    split_csv = "../data/" + args.database + "/splits/" + args.scenario + "/test" + args.database + ".csv"
    sample_ids = [r["sampleID"] for r in csv.DictReader(open(split_csv, encoding="utf-8"))]
    assert len(sample_ids) == len(loader.dataset), f"ids {len(sample_ids)} != dataset {len(loader.dataset)}"

    out = []
    outp = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with torch.no_grad():
        for k, (xs_pad, ilens, ys_pad, olens, refs) in enumerate(tqdm(loader, file=sys.stdout, mininterval=30)):
            result = speech2text(torch.squeeze(xs_pad, 0))
            cands = [norm(r[0]) for r in result[:NBEST]]
            out.append({"clip": sample_ids[k], "ref": norm(refs[0]), "cands": cands})
            if (k + 1) % 25 == 0:
                json.dump(out, open(outp, "w"), ensure_ascii=False)
    json.dump(out, open(outp, "w"), ensure_ascii=False)
    print(f"LISTO {len(out)} clips -> {outp}")
