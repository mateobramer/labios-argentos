#!/usr/bin/env python
"""Reconstruye el modelo personal a partir del delta que sube la GPU.

El fine-tune LoRA toca una fraccion de los tensores; bajar el merge completo
(1.1 GB) desperdicia ~10 min de red. La GPU sube solo los tensores que cambiaron
(<persona>_delta.pth) y aca se aplican sobre los pesos base locales.

Uso: aplicar_delta.py <delta.pth> <salida.pth> [base.pth]
     (base default: $VISPER_DIR/visper_vsr_base.pth, con VISPER_DIR=~/Desktop/visper)
"""
import os
import sys

import torch


def _sd(d):
    return d.get("state_dict", d) if isinstance(d, dict) else d


def main():
    delta_p, out_p = sys.argv[1], sys.argv[2]
    base_p = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.expanduser(os.environ.get("VISPER_DIR", "~/Desktop/visper")),
        "visper_vsr_base.pth")
    base = _sd(torch.load(base_p, map_location="cpu"))
    delta = _sd(torch.load(delta_p, map_location="cpu"))
    base.update(delta)
    torch.save(base, out_p)
    print(f"[delta] {len(delta)} tensores aplicados sobre base -> {out_p}")


if __name__ == "__main__":
    main()
