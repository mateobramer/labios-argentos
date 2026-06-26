"""
Fine-tuning del VSR español de Gimeno al rioplatense.

Enfoque (ver vsr_models/README.md): reusar el modelo ESPnet de Gimeno + TODO su pipeline
de datos ya probado (transforms, tokenizer char, collate `data_processing`) y solo agregar
un training loop propio sobre nuestros splits speaker-independent.

Reusa del repo de Gimeno (se pasa con --gimeno-repo):
  - src.Transforms: Compose([Normalise(0,250), Normalise(mean,std), CenterCrop((88,88))])
    -> mismisima normalizacion que uso el zero-shot (mean/std Rioplatense = 0.491/0.166).
  - src.utils.data_processing: collate que aplica transforms + tokeniza + padea el batch.
  - src.utils.get_tokenizer_converter: tokenizer char + converter del config.
  - espnet2 ASRTask.build_model_from_file: carga el ESPnetASRModel con los pesos del .pth.
    Su forward(speech, speech_lengths, text, text_lengths) -> (loss, stats, weight).

Uso (en la VM, env `vsr-factors`):
    python -m vsr_models.src.fine_tune \\
        --gimeno-repo ~/evaluating-end2end-spanish-lipreading \\
        --vsr-config  ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \\
        --load-vsr    ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \\
        --rois-root   ~/data/lip_rois \\
        --out         vsr_models/runs/ft01
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(DIR_MODULO, "splits")


class ClipsRioplatense(Dataset):
    """Devuelve (sampleID, lips (T,96,96) tensor, transcripcion) — el formato que espera
    `data_processing`. Los transforms/normalizacion los aplica el collate, no aca."""

    def __init__(self, split_csv, rois_root, max_frames=0):
        with open(split_csv, encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        # La self-attention del Conformer es O(T^2); saltear clips muy largos acota memoria.
        if max_frames > 0:
            items = [it for it in items if int(it["n_frames"]) <= max_frames]
        self.items = items
        self.rois_root = rois_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        npz = os.path.join(self.rois_root, it["titulo"], it["clip"] + ".npz")
        lips = torch.from_numpy(np.load(npz)["rois"])  # (T,96,96) uint8
        return it["clip"], lips, it["texto"]


def evaluar(asr_model, loader, device):
    asr_model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, xlens, y, ylens, _ in loader:
            out = asr_model(x.to(device), xlens.to(device), y.to(device), ylens.to(device))
            loss = out[0] if isinstance(out, (tuple, list)) else out["loss"]
            tot += float(loss) * x.size(0)
            n += x.size(0)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gimeno-repo", required=True)
    ap.add_argument("--vsr-config", required=True)
    ap.add_argument("--load-vsr", required=True)
    ap.add_argument("--rois-root", required=True, help="dir con <titulo>/<clip>.npz en la VM")
    ap.add_argument("--out", default=os.path.join(DIR_MODULO, "runs", "ft01"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)       # FT suave; tunear en val
    ap.add_argument("--paciencia", type=int, default=5)     # early stopping
    ap.add_argument("--accum", type=int, default=1, help="pasos de gradient accumulation")
    ap.add_argument("--max-frames", type=int, default=0, help="saltea clips mas largos (0=sin limite)")
    ap.add_argument("--smoke", action="store_true", help="1 batch train+val y salir (test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, os.path.expanduser(args.gimeno_repo))
    from src.utils import data_processing, get_tokenizer_converter
    from src.Transforms import Compose, Normalise, CenterCrop
    from src.tasks.asr import ASRTask  # ASRTask custom del repo: registra el frontend conv3dresnet18

    with open(args.vsr_config, encoding="utf-8") as f:
        cfg = argparse.Namespace(**yaml.safe_load(f))
    tokenizer, converter = get_tokenizer_converter(cfg.token_type, cfg.bpemodel, cfg.token_list)
    ignore_id = cfg.model_conf["ignore_id"]
    # MISMA normalizacion que el zero-shot (mean/std Rioplatense).
    transforms = Compose([Normalise(0.0, 250.0), Normalise(0.491, 0.166), CenterCrop((88, 88))])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model, _ = ASRTask.build_model_from_file(args.vsr_config, args.load_vsr, device)
    asr_model.to(device)

    # La capa Conformer del repo guarda stochastic_depth_rate como lista, lo que rompe el
    # forward en modo TRAIN (`rate > 0` sobre una lista); en eval no se evalua, por eso el
    # zero-shot andaba. Lo desactivamos (0.0): es regularizacion y para un FT chico conviene
    # apagarla igual.
    for layer in getattr(asr_model.encoder, "encoders", []):
        if hasattr(layer, "stochastic_depth_rate"):
            layer.stochastic_depth_rate = 0.0

    def collate(b):
        return data_processing(b, transforms, tokenizer, converter, ignore_id)

    def cargar(split):
        ds = ClipsRioplatense(os.path.join(SPLITS_DIR, f"{split}.csv"), args.rois_root, args.max_frames)
        return DataLoader(ds, batch_size=args.batch, shuffle=(split == "train"),
                          collate_fn=collate, num_workers=4)

    tr, va = cargar("train"), cargar("val")
    print(f"train={len(tr.dataset)}  val={len(va.dataset)}  batch={args.batch}  device={device}")

    opt = torch.optim.AdamW(asr_model.parameters(), lr=args.lr)

    if args.smoke:
        asr_model.train()
        x, xl, y, yl, _ = next(iter(tr))
        out = asr_model(x.to(device), xl.to(device), y.to(device), yl.to(device))
        loss = out[0] if isinstance(out, (tuple, list)) else out["loss"]
        loss.backward()
        opt.step()
        print(f"[smoke] forward+backward OK. loss={float(loss):.3f}  x={tuple(x.shape)}  y={tuple(y.shape)}")
        return

    mejor, sin_mejora = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        asr_model.train()
        opt.zero_grad()
        for i, (x, xl, y, yl, _) in enumerate(tr):
            out = asr_model(x.to(device), xl.to(device), y.to(device), yl.to(device))
            loss = out[0] if isinstance(out, (tuple, list)) else out["loss"]
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(asr_model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
            if i % 100 == 0:
                print(f"ep{ep} it{i}/{len(tr)} loss={float(loss):.3f}", flush=True)
        val = evaluar(asr_model, va, device)
        print(f"== ep{ep} val_loss={val:.4f} ==", flush=True)
        if val < mejor:
            mejor, sin_mejora = val, 0
            torch.save(asr_model.state_dict(), os.path.join(args.out, "best.pth"))
            print(f"  nuevo mejor ({val:.4f}) -> best.pth")
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print(f"early stopping (sin mejora hace {args.paciencia} epochs)")
                break
    print(f"\nListo. Mejor val_loss={mejor:.4f}. best.pth en {args.out}. "
          f"Evaluar WER con vsr_main.py sobre el split test.")


if __name__ == "__main__":
    main()
