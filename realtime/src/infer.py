"""
Inferencia VSR para el servicio: ROI labial (T,96,96) -> texto.

Reusa el modelo y el decode de Gimeno (`Speech2Text`, beam search CTC/attention de
ESPnet) tal cual los usa el `vsr_main.py` del zero-shot, pero cargando el checkpoint
fine-tuneado al rioplatense (`ft03_best.pth`).

El modelo se construye UNA vez (es caro) y se reusa entre requests. Pensado para correr
en CPU en el Mac: con beam 10 da RTF < 0.5 (más rápido que tiempo real) para clips cortos.

Variables de entorno:
    GIMENO_REPO   ruta al repo de Gimeno (default ~/evaluating-end2end-spanish-lipreading)
    VSR_CONFIG    config yaml         (default realtime/models/vsr_config.yaml)
    VSR_CKPT      checkpoint .pth     (default realtime/models/ft03_best.pth)
    VSR_DEVICE    cpu|mps|cuda        (default cpu)
    VSR_BEAM      beam size           (default 10)
"""

import os
import sys

import numpy as np
import torch
import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELOS = os.path.join(os.path.dirname(_DIR), "models")

GIMENO_REPO = os.path.expanduser(os.environ.get("GIMENO_REPO", "~/evaluating-end2end-spanish-lipreading"))
VSR_CONFIG = os.environ.get("VSR_CONFIG", os.path.join(_MODELOS, "vsr_config.yaml"))
VSR_CKPT = os.environ.get("VSR_CKPT", os.path.join(_MODELOS, "ft03_best.pth"))
VSR_DEVICE = os.environ.get("VSR_DEVICE", "cpu")
VSR_BEAM = int(os.environ.get("VSR_BEAM", "10"))
VSR_NBEST = int(os.environ.get("VSR_NBEST", "5"))  # candidatas que devuelve el beam (<= beam)


class VSRInfer:
    """Carga el modelo una vez; `transcribir(rois)` devuelve la hipótesis de texto."""

    def __init__(self, config=VSR_CONFIG, ckpt=VSR_CKPT, device=VSR_DEVICE, beam=VSR_BEAM,
                 nbest=VSR_NBEST):
        if GIMENO_REPO not in sys.path:
            sys.path.insert(0, GIMENO_REPO)
        from src.bin.asr_inference import Speech2Text
        from src.Transforms import CenterCrop, Compose, Normalise

        with open(config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        inf_conf = dict(cfg["inference_conf"])
        inf_conf["beam_size"] = beam
        inf_conf["nbest"] = min(nbest, beam)   # pedirle al beam las top-N candidatas
        inf_conf.pop("lm_weight", None)  # sin LM (el LM peninsular no ayuda al rioplatense)

        self.device = device
        # Misma normalización que el zero-shot/fine-tune: /250 + mean/std LIP-RTVE, center crop.
        self.transforms = Compose([
            Normalise(0.0, 250.0),
            Normalise(0.491, 0.166),
            CenterCrop((88, 88)),
        ])
        self.s2t = Speech2Text(
            asr_train_config=config,
            asr_model_file=ckpt,
            lm_train_config=None,
            lm_file=None,
            device=device,
            **inf_conf,
        )

    def _decode(self, rois):
        x = torch.from_numpy(np.ascontiguousarray(rois)).float()  # (T,96,96)
        x = self.transforms(x)                                    # (T,88,88)
        with torch.no_grad():
            return self.s2t(x)  # lista de (text, token, token_int, hyp), largo = nbest

    def transcribir(self, rois):
        """rois (T,96,96) uint8 -> str: la 1-best (texto en minúsculas, sin puntuación)."""
        return self._decode(rois)[0][0].strip()

    def transcribir_nbest(self, rois):
        """rois (T,96,96) uint8 -> list[str]: las top-N candidatas del beam (mejor primero)."""
        return [r[0].strip() for r in self._decode(rois)]


_SINGLETON = None


def get_infer():
    """Devuelve la instancia única de VSRInfer (la crea en el primer llamado)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = VSRInfer()
    return _SINGLETON
