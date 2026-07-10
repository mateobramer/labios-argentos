#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Servidor de inferencia ViSpeR para la demo push-to-talk.
Carga el modelo UNA vez y despues lee rutas de .npz por stdin (una por linea),
devolviendo la transcripcion por stdout (una por linea). Asi cada apreton de la
barra es rapido (no recarga el modelo de 1.1GB cada vez).

Corre en el env `visper`:
  ~/miniconda3/envs/visper/bin/python infer_server.py

Variables de entorno:
  VSR_BEAM   beam-size (default 3 = punto de Pareto, ver experiments/09).
  VSR_QWEN   si esta (=1), corrige con qwen n-best rescoring (Ollama local).
             Fuerza beam>=5 (el rescoring necesita >=5 candidatas). Agrega ~1.24s/frase.
  VSR_QMODEL modelo de Ollama (default qwen3:4b-instruct-2507-q4_K_M).
  VSR_CKPT   state_dict alternativo (p.ej. modelos/personal/<nombre>.pth = LoRA mergeado
             de calibracion al hablante, ver experiments/10). Vacio = ViSpeR base.
"""
import os, sys, re, json, unicodedata
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # ops sin kernel MPS caen a CPU (antes de importar torch)
print("[infer] importando librerias + modelo ViSpeR (primera vez ~20-40s, NO cortar)...",
      file=sys.stderr, flush=True)
REPO = os.path.expanduser("~/Desktop/visper")
sys.path.insert(0, REPO)
import numpy as np, torch
from datamodule.transforms import VideoTransform
from lightning_vsr import ModelModule, get_beam_search_decoder
from visper_zeroshot import build_cfg   # reusa la config exacta del zero-shot

LANG = 4  # <es>
USE_QWEN = os.environ.get("VSR_QWEN", "") not in ("", "0", "false", "False")
NBEST = 5   # candidatas maximas que se le pasan al LLM (si el beam da menos, se usan las que haya)
# beam default: 3 sin qwen (punto de Pareto, ver experiments/09); 5 con qwen (mejor techo de rescoring).
# Se respeta VSR_BEAM si lo seteas (con qwen, minimo 2: el rescoring necesita >=2 candidatas).
BEAM = int(os.environ.get("VSR_BEAM", "5" if USE_QWEN else "3"))
QMODEL = os.environ.get("VSR_QMODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA = "http://127.0.0.1:11434"
import urllib.request
RT = {"qwen": USE_QWEN}   # toggleable en runtime: linea de control "::qwen 0|1" por stdin
if USE_QWEN:
    BEAM = max(BEAM, 2)

SYS_RESC = ("Te doy VARIAS transcripciones candidatas de la MISMA frase (lectura de labios, rioplatense). "
            "Se complementan: la palabra correcta suele estar en alguna. Combinalas para la frase mas probable, "
            "sin inventar. Responde SOLO la frase, una linea.")

def norm(s):
    s = s.lower().strip().replace("ñ", "\x00")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\x00", "ñ")
    s = re.sub(r"[^a-z0-9ñ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def qwen_rescore(cands):
    """Manda las candidatas a Ollama y devuelve la frase elegida. Si falla, cae a la 1-best."""
    user = "Candidatas:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands)) + "\n\nFrase:"
    body = {"model": QMODEL, "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0.0, "num_predict": 256},
            "messages": [{"role": "system", "content": SYS_RESC}, {"role": "user", "content": user}]}
    try:
        req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            o = json.loads(r.read().decode())
        txt = re.sub(r"<think>.*?</think>", " ", o.get("message", {}).get("content", ""), flags=re.DOTALL)
        out = norm(txt)
        return out if out else cands[0]
    except Exception as e:
        print(f"[infer] qwen fallo ({e}), uso 1-best", file=sys.stderr, flush=True)
        return cands[0]

def main():
    print(f"[infer] cargando ViSpeR... (qwen={'ON' if USE_QWEN else 'off'})", file=sys.stderr, flush=True)
    mm = ModelModule(build_cfg())
    ckpt_name = "base"
    ckpt = os.environ.get("VSR_CKPT", "")
    if ckpt:
        sd = torch.load(os.path.expanduser(ckpt), map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        mm.model.load_state_dict(sd)
        ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
        print(f"[infer] checkpoint personalizado: {ckpt_name}", file=sys.stderr, flush=True)
    # beam configurable (default del repo es 40, puro desperdicio). Ver experiments/09.
    mm.beam_search = get_beam_search_decoder(mm.model, mm.token_list, ctc_weight=0.1, beam_size=BEAM)
    # HIBRIDO validado (experiments/09): ENCODER en MPS (3.4x, transcripciones identicas 100/100)
    # + beam en CPU (el beam de espnet SI rompe en MPS por device-mismatch, no moverlo).
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mm.eval(); mm.to(dev)
    enc_dev = dev
    if dev == "cpu" and torch.backends.mps.is_available() and os.environ.get("VSR_MPS", "1") != "0":
        enc_dev = "mps"
        mm.model.encoder.to(enc_dev)
    vt = VideoTransform(subset="test")
    from espnet.asr.asr_utils import add_results_to_json as _arj
    print(f"[infer] listo (encoder={enc_dev}, beam={BEAM} en {dev}, qwen={'ON' if USE_QWEN else 'off'}). Esperando npz...",
          file=sys.stderr, flush=True)
    # linea CONFIG antes de READY: los clientes que solo esperan READY la ignoran
    print("CONFIG " + json.dumps({"encoder": enc_dev, "beam": BEAM, "qwen": RT["qwen"],
                                  "modelo": ckpt_name}), flush=True)
    print("READY", flush=True)   # senal para el cliente

    for line in sys.stdin:
        npzp = line.strip()
        if not npzp:
            continue
        if npzp.startswith("::"):                 # linea de control (responde SIEMPRE 1 linea)
            if npzp.startswith("::qwen"):
                RT["qwen"] = npzp.split()[-1] in ("1", "on", "true")
            print(f"::ok qwen={int(RT['qwen'])}", flush=True)
            continue
        try:
            rois = np.load(npzp)["rois"]                                   # (T,96,96) uint8
            v = torch.tensor(rois).unsqueeze(1).repeat(1, 3, 1, 1).float() # (T,3,96,96)
            data = vt(v)
            with torch.no_grad():
                enc, _ = mm.model.encoder(data.unsqueeze(0).to(enc_dev), None)
                enc = enc.squeeze(0).to(dev)             # features al CPU para el beam
                hyps = mm.beam_search(enc, lang=LANG)
                if not RT["qwen"]:
                    hyp = _arj([hyps[0].asdict()], mm.token_list, LANG).replace("▁", " ").replace("<eos>", "")
                    print(norm(hyp), flush=True)
                else:
                    cands = [norm(_arj([h.asdict()], mm.token_list, LANG)
                                  .replace("▁", " ").replace("<eos>", ""))
                             for h in hyps[:min(len(hyps), NBEST)]]
                    print(qwen_rescore(cands), flush=True)
        except Exception as e:
            print(f"__ERROR__ {e}", flush=True)

if __name__ == "__main__":
    main()
