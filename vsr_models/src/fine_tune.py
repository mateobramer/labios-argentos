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


class RandomCrop(object):
    """Crop aleatorio de (th,tw) — MISMA posicion para todos los frames del clip
    (augment espacial estandar VSR). Reemplaza a CenterCrop solo en train."""

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, video_data):
        frames, h, w = video_data.shape
        th, tw = self.crop_size
        i = int(torch.randint(0, h - th + 1, (1,)).item())
        j = int(torch.randint(0, w - tw + 1, (1,)).item())
        return video_data[:, i:i + th, j:j + tw]


class HorizontalFlip(object):
    """Flip horizontal del clip entero con probabilidad p (augment estandar VSR)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video_data):
        if float(torch.rand(1).item()) < self.p:
            return torch.flip(video_data, dims=[2])
        return video_data


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
    ap.add_argument("--freeze", default=None,
                    help="congela (requires_grad=False) los params bajo este modulo top-level, ej: frontend")
    ap.add_argument("--augment", action="store_true",
                    help="augment espacial SOLO en train: RandomCrop(88)+HFlip(0.5); val/test quedan en CenterCrop")
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
    # MISMA normalizacion que el zero-shot (mean/std Rioplatense). Eval => CenterCrop (sin tocar).
    base_norm = [Normalise(0.0, 250.0), Normalise(0.491, 0.166)]
    eval_tf = Compose(base_norm + [CenterCrop((88, 88))])
    if args.augment:
        # config v2 (reconstruccion estandar VSR; la receta original de v2 vivia en su train.log,
        # hoy inaccesible): RandomCrop + flip horizontal SOLO en train. val/test intactos.
        train_tf = Compose(base_norm + [RandomCrop((88, 88)), HorizontalFlip(0.5)])
        print("[augment] train: RandomCrop(88)+HFlip(0.5); val/test: CenterCrop(88)", flush=True)
    else:
        train_tf = eval_tf

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

    # -- freeze opcional (config v2: congela el frontend Conv3D-ResNet18) --
    if args.freeze:
        nf = 0
        for name, p in asr_model.named_parameters():
            if name == args.freeze or name.startswith(args.freeze + "."):
                p.requires_grad = False
                nf += 1
        if nf == 0:
            print(f"[freeze] OJO: '{args.freeze}' no matcheo ningun parametro", flush=True)
        else:
            print(f"[freeze] '{args.freeze}': {nf} params congelados", flush=True)

    def collate_with(tf):
        def _c(b):
            return data_processing(b, tf, tokenizer, converter, ignore_id)
        return _c

    def cargar(split):
        ds = ClipsRioplatense(os.path.join(SPLITS_DIR, f"{split}.csv"), args.rois_root, args.max_frames)
        tf = train_tf if split == "train" else eval_tf
        return DataLoader(ds, batch_size=args.batch, shuffle=(split == "train"),
                          collate_fn=collate_with(tf), num_workers=4)

    tr, va = cargar("train"), cargar("val")
    print(f"train={len(tr.dataset)}  val={len(va.dataset)}  batch={args.batch}  device={device}")

    params = [p for p in asr_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

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
