"""
Fine-tuning del VSR español de Gimeno al rioplatense.

Enfoque (ver vsr/README.md): reusar el modelo ESPnet de Gimeno + TODO su pipeline
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
    python -m vsr.src.fine_tune \\
        --gimeno-repo ~/evaluating-end2end-spanish-lipreading \\
        --vsr-config  ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \\
        --load-vsr    ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth \\
        --rois-root   ~/data/lip_rois \\
        --splits-dir  vsr/splits \\
        --transcripts-root "" \\
        --out         vsr/runs/ft01
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

DIR_MODULO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(DIR_MODULO, "splits")


def set_seed(seed):
    """Fija el seed en random/numpy/torch para comparaciones head-to-head reproducibles
    (ej: base LIP-RTVE vs multilingue con el MISMO orden de datos y dropout)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RandomCrop:
    """(T,H,W) -> (T,size,size) con offset aleatorio. Data augmentation para train."""
    def __init__(self, size):
        self.size = size

    def __call__(self, v):
        _, h, w = v.shape
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        return v[:, top:top + self.size, left:left + self.size]


class RandomFlip:
    """Flip horizontal con prob p (la boca es ~simetrica). Augmentation para train."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, v):
        return torch.flip(v, dims=[2]) if random.random() < self.p else v


class ClipsRioplatense(Dataset):
    """Devuelve (sampleID, lips (T,96,96) tensor, transcripcion) — el formato que espera
    `data_processing`. Los transforms/normalizacion los aplica el collate, no aca."""

    def __init__(self, split_csv, rois_root, max_frames=0, transcripts_root=""):
        with open(split_csv, encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        # La self-attention del Conformer es O(T^2); saltear clips muy largos acota memoria.
        if max_frames > 0:
            items = [it for it in items if int(it["n_frames"]) <= max_frames]
        self.items = items
        self.rois_root = rois_root
        self.transcripts_root = transcripts_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        npz = os.path.join(self.rois_root, it["titulo"], it["clip"] + ".npz")
        texto = it["texto"]
        if self.transcripts_root:
            with open(transcript_txt_path(self.transcripts_root, it["titulo"], it["clip"]),
                      encoding="utf-8") as f:
                texto = f.read().strip()
        lips = torch.from_numpy(np.load(npz)["rois"])  # (T,96,96) uint8
        return it["clip"], lips, texto


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


def split_csv_path(splits_dir, split):
    return os.path.normpath(os.path.join(splits_dir, f"{split}.csv"))


def transcript_txt_path(transcripts_root, titulo, clip):
    return os.path.normpath(os.path.join(transcripts_root, titulo, f"{clip}.txt"))


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gimeno-repo", required=True)
    ap.add_argument("--vsr-config", required=True)
    ap.add_argument("--load-vsr", required=True)
    ap.add_argument("--rois-root", required=True, help="dir con <titulo>/<clip>.npz en la VM")
    ap.add_argument(
        "--splits-dir",
        default=SPLITS_DIR,
        help="dir con train.csv y val.csv (default: vsr/splits)",
    )
    ap.add_argument(
        "--transcripts-root",
        default="",
        help="dir opcional con <titulo>/<clip>.txt; default usa la columna texto del split",
    )
    ap.add_argument("--out", default=os.path.join(DIR_MODULO, "runs", "ft01"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)       # FT suave; tunear en val
    ap.add_argument("--paciencia", type=int, default=5)     # early stopping
    ap.add_argument("--accum", type=int, default=1, help="pasos de gradient accumulation")
    ap.add_argument("--max-frames", type=int, default=0, help="saltea clips mas largos (0=sin limite)")
    ap.add_argument("--freeze", default="", help="modulos a congelar, coma-separados (ej: frontend o frontend,encoder)")
    ap.add_argument("--augment", action="store_true", help="data augmentation en train (random crop + flip)")
    ap.add_argument("--smoke", action="store_true", help="1 batch train+val y salir (test)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="seed para reproducibilidad (comparacion head-to-head entre bases)")
    return ap


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    print(f"[seed] {args.seed}", flush=True)

    sys.path.insert(0, os.path.expanduser(args.gimeno_repo))
    from src.utils import data_processing, get_tokenizer_converter
    from src.Transforms import Compose, Normalise, CenterCrop
    from src.tasks.asr import ASRTask  # ASRTask custom del repo: registra el frontend conv3dresnet18

    with open(args.vsr_config, encoding="utf-8") as f:
        cfg = argparse.Namespace(**yaml.safe_load(f))
    tokenizer, converter = get_tokenizer_converter(cfg.token_type, cfg.bpemodel, cfg.token_list)
    ignore_id = cfg.model_conf["ignore_id"]
    # MISMA normalizacion que el zero-shot (/250 + mean/std Rioplatense). Val/eval: center crop.
    norm = [Normalise(0.0, 250.0), Normalise(0.491, 0.166)]
    transforms_val = Compose(norm + [CenterCrop((88, 88))])
    # Train con --augment: random crop + flip horizontal -> "multiplican" los datos, menos overfit.
    transforms_train = Compose(norm + [RandomCrop(88), RandomFlip(0.5)]) if args.augment else transforms_val

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model, _ = ASRTask.build_model_from_file(args.vsr_config, args.load_vsr, device)
    asr_model.to(device)

    # Bug del repo: stochastic_depth_rate como lista rompe el forward en TRAIN (no en eval, por
    # eso el zero-shot andaba). Lo desactivamos (0.0): regularizacion que para un FT chico igual conviene apagar.
    for layer in getattr(asr_model.encoder, "encoders", []):
        if hasattr(layer, "stochastic_depth_rate"):
            layer.stochastic_depth_rate = 0.0

    # Congelar modulos (ej: "frontend" o "frontend,encoder") -> menos params que ajustar -> menos overfit.
    congelar = {s.strip() for s in args.freeze.split(",") if s.strip()}
    for n, mod in asr_model.named_children():
        if n in congelar:
            for p in mod.parameters():
                p.requires_grad = False
    entrenables = sum(p.numel() for p in asr_model.parameters() if p.requires_grad) / 1e6

    def cargar(split, transforms):
        ds = ClipsRioplatense(
            split_csv_path(args.splits_dir, split),
            args.rois_root,
            args.max_frames,
            args.transcripts_root,
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=(split == "train"),
                          collate_fn=lambda b: data_processing(b, transforms, tokenizer, converter, ignore_id),
                          num_workers=4)

    tr = cargar("train", transforms_train)
    va = cargar("val", transforms_val)
    print(f"train={len(tr.dataset)} val={len(va.dataset)} | congelados={sorted(congelar) or 'ninguno'} "
          f"entrenables={entrenables:.1f}M | augment={args.augment} lr={args.lr} "
          f"splits_dir={args.splits_dir} transcripts_root={args.transcripts_root or 'split_csv'} "
          f"device={device}")

    opt = torch.optim.AdamW([p for p in asr_model.parameters() if p.requires_grad], lr=args.lr)

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
                torch.nn.utils.clip_grad_norm_(params, 5.0)
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
