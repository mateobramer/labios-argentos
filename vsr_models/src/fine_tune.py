"""
Fine-tuning del VSR español de Gimeno al rioplatense.  [BORRADOR — validar en la VM]

Enfoque (ver vsr_models/README.md): reusar la carga del modelo ESPnet de Gimeno
(`Speech2Text` -> `.asr_model`, que es un `ESPnetASRModel` con forward que devuelve loss)
y agregarle un training loop propio sobre nuestros splits speaker-independent.

Lo que ESTE archivo deja resuelto (probado offline): leer los splits, armar batches de
ROIs (T,96,96)->CenterCrop(88)+Normalise, el loop train/val, optimizer, early-stopping y
checkpointing.  Lo que hay que CONFIRMAR en la VM contra el repo de Gimeno esta marcado
con `# TODO[VM]` (shapes exactas del frontend visual y API del tokenizer/converter).

Uso (en la VM, env `vsr-factors`, con el repo de Gimeno en PYTHONPATH):
    python -m vsr_models.src.fine_tune \\
        --gimeno-repo ~/evaluating-end2end-spanish-lipreading \\
        --vsr-config <config.yaml de Zenodo> \\
        --load-vsr   <.../VSR/vsr-liprtve-si.pth> \\
        --rois-root  ~/data/lip_rois \\
        --out        vsr_models/runs/ft01
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 96x96 -> CenterCrop(88) + Normalise; mean/std de LIP-RTVE (parche Rioplatense del eval).
CROP = 88
MEAN, STD = 0.491, 0.166

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ_REPO = os.path.dirname(DIR_MODULO)
SPLITS_DIR = os.path.join(DIR_MODULO, "splits")


# --------------------------------------------------------------------------- datos

def _center_crop(rois):
    """(T,96,96) uint8 -> (T,88,88) float32 normalizado, recorte centrado."""
    t, h, w = rois.shape
    top, left = (h - CROP) // 2, (w - CROP) // 2
    x = rois[:, top:top + CROP, left:left + CROP].astype(np.float32) / 255.0
    return (x - MEAN) / STD


class ClipsRioplatense(Dataset):
    """Lee un split CSV (split,spk,titulo,clip,n_frames,texto,npz) y devuelve (rois, texto)."""

    def __init__(self, split_csv, rois_root):
        with open(split_csv, encoding="utf-8") as f:
            self.items = list(csv.DictReader(f))
        self.rois_root = rois_root  # donde viven los .npz en la VM (bajados del bucket)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        # el csv guarda npz relativo al repo; en la VM remapeamos a --rois-root
        npz = os.path.join(self.rois_root, it["titulo"], it["clip"] + ".npz")
        rois = np.load(npz)["rois"]            # (T,96,96) uint8
        return _center_crop(rois), it["texto"]


def hacer_collate(tokenizar):
    """Arma el batch: padea ROIs en T y tokeniza el texto. tokenizar: str -> LongTensor(ids)."""
    def collate(batch):
        rois, textos = zip(*batch)
        lens = torch.tensor([r.shape[0] for r in rois], dtype=torch.long)
        tmax = int(lens.max())
        # (B, Tmax, 88, 88)
        x = torch.zeros(len(rois), tmax, CROP, CROP, dtype=torch.float32)
        for k, r in enumerate(rois):
            x[k, : r.shape[0]] = torch.from_numpy(r)
        ids = [tokenizar(t) for t in textos]
        ylens = torch.tensor([len(y) for y in ids], dtype=torch.long)
        y = torch.full((len(ids), int(ylens.max())), -1, dtype=torch.long)  # pad -1 (ignore)
        for k, t in enumerate(ids):
            y[k, : len(t)] = t
        return x, lens, y, ylens
    return collate


# ------------------------------------------------------------------- modelo (Gimeno)

def cargar_gimeno(args):
    """Reusa la carga de inferencia de Gimeno. Devuelve (asr_model, tokenizar)."""
    sys.path.insert(0, os.path.expanduser(args.gimeno_repo))
    from src.bin.asr_inference import Speech2Text  # noqa: la misma del zero-shot

    s2t = Speech2Text(asr_train_config=args.vsr_config, asr_model_file=args.load_vsr)
    asr_model = s2t.asr_model  # ESPnetASRModel: forward(speech, speech_lengths, text, text_lengths) -> (loss, stats, weight)

    # TODO[VM]: confirmar la API de tokenizacion char-level. En ESPnet suele ser:
    #   ids = s2t.converter.tokens2ids(s2t.tokenizer.text2tokens(texto))
    def tokenizar(texto):
        ids = s2t.converter.tokens2ids(s2t.tokenizer.text2tokens(texto))
        return torch.tensor(ids, dtype=torch.long)

    return asr_model, tokenizar


def forward_loss(asr_model, x, xlens, y, ylens, device):
    """Un paso forward que devuelve el loss escalar."""
    # TODO[VM]: confirmar shape que espera el frontend visual. En Auto-AVSR/ESPnet el
    # Conv3D toma (B, T, H, W) o (B, 1, T, H, W). Ajustar aca si hace falta un unsqueeze.
    x = x.to(device)
    out = asr_model(x, xlens.to(device), y.to(device), ylens.to(device))
    loss = out[0] if isinstance(out, (tuple, list)) else out["loss"]
    return loss


# --------------------------------------------------------------------------- loop

def evaluar(asr_model, loader, device):
    asr_model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, xlens, y, ylens in loader:
            loss = forward_loss(asr_model, x, xlens, y, ylens, device)
            tot += float(loss) * x.size(0)
            n += x.size(0)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gimeno-repo", required=True)
    ap.add_argument("--vsr-config", required=True)
    ap.add_argument("--load-vsr", required=True)
    ap.add_argument("--rois-root", required=True, help="dir con los .npz en la VM (del bucket)")
    ap.add_argument("--out", default=os.path.join(DIR_MODULO, "runs", "ft01"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)        # FT suave; tunear en val
    ap.add_argument("--paciencia", type=int, default=5)      # early stopping
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model, tokenizar = cargar_gimeno(args)
    asr_model.to(device)

    collate = hacer_collate(tokenizar)
    tr = DataLoader(ClipsRioplatense(os.path.join(SPLITS_DIR, "train.csv"), args.rois_root),
                    batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=4)
    va = DataLoader(ClipsRioplatense(os.path.join(SPLITS_DIR, "val.csv"), args.rois_root),
                    batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=4)

    opt = torch.optim.AdamW(asr_model.parameters(), lr=args.lr)
    mejor, sin_mejora = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        asr_model.train()
        for i, (x, xlens, y, ylens) in enumerate(tr):
            opt.zero_grad()
            loss = forward_loss(asr_model, x, xlens, y, ylens, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(asr_model.parameters(), 5.0)
            opt.step()
            if i % 50 == 0:
                print(f"ep{ep} it{i} loss={float(loss):.3f}")
        val = evaluar(asr_model, va, device)
        print(f"== ep{ep} val_loss={val:.4f} ==")
        if val < mejor:
            mejor, sin_mejora = val, 0
            torch.save(asr_model.state_dict(), os.path.join(args.out, "best.pth"))
            print(f"  nuevo mejor ({val:.4f}) -> best.pth")
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print(f"early stopping (sin mejora hace {args.paciencia} epochs)")
                break
    print(f"\nListo. Mejor val_loss={mejor:.4f}. Evaluar WER con el evaluador de Gimeno "
          f"sobre test.csv usando best.pth.")


if __name__ == "__main__":
    main()
