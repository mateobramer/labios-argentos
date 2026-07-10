"""
Servicio de VSR en vivo (demo kiosko).

Flujo de un request:
    navegador graba un clip (push-to-talk) -> POST /transcribe (video webm/mp4)
      -> preprocess_live: cuadros -> ROI labial (T,96,96) a 25 fps
      -> infer: ROI -> texto (modelo ft03 + beam search de ESPnet)
      -> JSON {texto, n_frames, ratio_deteccion, rtf, ...}

Diseño "por clips" (lo charlado): el modelo es bidireccional, así que corre sobre la
utterance completa (lookahead acotado al clip), sin cirugía al modelo. Push-to-talk evita
el problema de segmentar sin audio. El VAD visual y el corrector LLM quedan para después.

Correr (desde la raíz del repo, env `realtime`):
    python -m realtime.src.server
    # luego abrir http://localhost:8000
"""

import os
import sys
import tempfile
import time

# La raíz del repo en sys.path para importar `visual_preprocessing` y `realtime`.
_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from realtime.src.corrector import get_corrector
from realtime.src.infer import get_infer
from realtime.src.preprocess_live import rois_desde_video

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="Labios Argentos — VSR en vivo")


@app.on_event("startup")
def _calentar():
    # Construir el modelo al arrancar (no en el primer request) para que la demo no
    # tenga un primer turno lento.
    print(">>> cargando modelo VSR ...", flush=True)
    t0 = time.time()
    get_infer()
    corr = get_corrector()
    estado = f"corrector LLM ON ({corr.modelo})" if corr.activo else "corrector LLM OFF (backend no disponible)"
    print(f">>> modelo listo en {time.time() - t0:.1f}s · {estado}. Abrí http://localhost:8000", flush=True)


@app.get("/")
def index():
    return FileResponse(os.path.join(_WEB, "index.html"))


@app.post("/transcribe")
def transcribe(clip: UploadFile = File(...)):
    sufijo = os.path.splitext(clip.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(clip.file.read())
        ruta = tmp.name
    try:
        t0 = time.time()
        rois, ratio, n_leidos, fps = rois_desde_video(ruta)
        t_pre = time.time() - t0

        if rois is None or len(rois) == 0:
            return JSONResponse({
                "ok": False,
                "error": "no_cara",
                "mensaje": "No se detectó una cara frontal estable. Acercate y mirá a la cámara.",
                "ratio_deteccion": round(ratio, 3),
                "n_frames_leidos": n_leidos,
            })

        t1 = time.time()
        crudo = get_infer().transcribir(rois)
        t_inf = time.time() - t1

        t2 = time.time()
        corregido = get_corrector().corregir(crudo)
        t_corr = time.time() - t2

        dur_s = len(rois) / 25.0
        return JSONResponse({
            "ok": True,
            "texto": corregido,            # lo que se muestra (corregido si hay LLM, si no = crudo)
            "texto_crudo": crudo,          # salida directa del VSR (para mostrar el antes/después)
            "corregido": corregido != crudo,
            "n_frames": int(len(rois)),
            "duracion_s": round(dur_s, 2),
            "ratio_deteccion": round(ratio, 3),
            "t_preproc_s": round(t_pre, 2),
            "t_infer_s": round(t_inf, 2),
            "t_corrector_s": round(t_corr, 2),
            "rtf": round((t_pre + t_inf + t_corr) / dur_s, 2) if dur_s > 0 else None,
        })
    finally:
        os.unlink(ruta)


# Estáticos (por si se agregan assets); el index se sirve en "/".
if os.path.isdir(_WEB):
    app.mount("/static", StaticFiles(directory=_WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
