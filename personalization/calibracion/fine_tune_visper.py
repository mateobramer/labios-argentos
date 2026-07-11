#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune LoRA de ViSpeR para calibracion al hablante (receta A de
docs/experiments/10: r16/alfa32 en las atenciones, lr 1e-4, augment, early-stop).
Reconstruccion del script perdido con la imagen labios-img-visper; corre en la GPU
fija dentro de ~/visper (o /root/visper) con los pesos base al lado.

Uso (lo invoca el daemon):
  python fine_tune_visper.py --train-csv .../cal_train.csv --val-csv .../cal_val.csv \
      --data-root .../data --out .../cal_<p> --lora --lora-r 16 --lora-alpha 32 \
      --augment --lr 1e-4 --accum 8 --epochs 20 --paciencia 5 --max-frames 400

Salida: <out>/best.pth = state_dict del modelo MERGEADO (mismo formato que carga
demo/infer_server.py via VSR_CKPT).
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamodule.transforms import VideoTransform
from lightning_vsr import ModelModule
from visper_zeroshot import build_cfg

LANG = "<es>"


def leer_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cargar_clip(root, fila, vt, max_frames):
    rois = np.load(os.path.join(root, fila["npz"]))["rois"]      # (T,96,96) uint8
    if max_frames and len(rois) > max_frames:
        return None
    v = torch.tensor(rois).unsqueeze(1).repeat(1, 3, 1, 1).float()  # (T,3,96,96)
    return vt(v)                                                  # (T,1,88,88) normalizado


def lote(mm, root, fila, vt, max_frames, device):
    x = cargar_clip(root, fila, vt, max_frames)
    if x is None:
        return None
    y = mm.text_transform.tokenize(fila["texto"].strip().lower(), language=LANG)
    x = x.unsqueeze(0).to(device)          # (1,T,1,88,88): mismo shape que usa infer_server
    lens = torch.tensor([x.shape[1]], device=device)
    return x, lens, y.unsqueeze(0).to(device)


def perdida_val(mm, filas, root, vt, max_frames, device):
    mm.model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for fila in filas:
            b = lote(mm, root, fila, vt, max_frames, device)
            if b is None:
                continue
            loss, _, _, _ = mm.model(*b, True)
            tot += float(loss); n += 1
    mm.model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--paciencia", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=400)
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mm = ModelModule(build_cfg())                                  # carga visper_vsr_base.pth
    mm.model.to(device)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                          target_modules=["linear_q", "linear_k", "linear_v", "linear_out"])
        mm.model = get_peft_model(mm.model, lcfg)
        mm.model.print_trainable_parameters()

    train = leer_csv(args.train_csv)
    val = leer_csv(args.val_csv)
    vt_train = VideoTransform(subset="train" if args.augment else "val")
    vt_val = VideoTransform(subset="val")
    opt = torch.optim.AdamW((p for p in mm.model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)

    mejor, sin_mejora = float("inf"), 0
    mm.model.train()
    for ep in range(1, args.epochs + 1):
        random.shuffle(train)
        tot, n = 0.0, 0
        opt.zero_grad()
        for i, fila in enumerate(train, 1):
            b = lote(mm, args.data_root, fila, vt_train, args.max_frames, device)
            if b is None:
                continue
            loss, _, _, _ = mm.model(*b, True)
            (loss / args.accum).backward()
            tot += float(loss); n += 1
            if i % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(mm.model.parameters(), 5.0)
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        vl = perdida_val(mm, val, args.data_root, vt_val, args.max_frames, device)
        print(f"[ep {ep:02d}] train_loss={tot/max(n,1):.3f} val_loss={vl:.3f}", flush=True)
        if vl < mejor - 1e-4:
            mejor, sin_mejora = vl, 0
            if args.lora:   # mergear una COPIA: el entrenamiento sigue intacto
                import copy
                nucleo = copy.deepcopy(mm.model).merge_and_unload()
            else:
                nucleo = mm.model
            torch.save({k: v.cpu() for k, v in nucleo.state_dict().items()},
                       os.path.join(args.out, "best.pth"))
            if args.lora:
                del nucleo
            print("  mejor val: guardo best.pth", flush=True)
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print(f"early stop en ep {ep} (paciencia {args.paciencia})", flush=True)
                break
    print(f"FIN mejor_val={mejor:.3f}", flush=True)


if __name__ == "__main__":
    main()
