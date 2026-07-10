"""
Servidor de captura para la adaptación al hablante.

Muestra las frases de realtime/adaptacion/frases_grabacion.md una por una; el usuario
graba cada frase con push-to-talk (webcam del navegador) y acá se guarda el par
    realtime/adaptacion/grabaciones/frase_NNN.mp4  (H.264 25fps, con audio)
    realtime/adaptacion/grabaciones/frase_NNN.txt  (texto normalizado como el dataset)
Reanudable: al abrir salta a la primera frase sin grabar; se puede regrabar cualquiera.

NO carga el modelo VSR (es solo captura). Correr (desde la raíz del repo, env realtime):
    python -m realtime.src.grabar_server
    # abrir http://localhost:8001
"""

import os
import re
import subprocess
import sys
import tempfile

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from unidecode import unidecode

_ADAPT = os.path.join(_RAIZ, "realtime", "adaptacion")
_FRASES_MD = os.path.join(_ADAPT, "frases_grabacion.md")
_GRAB = os.path.join(_ADAPT, "grabaciones")
_WEB = os.path.join(_RAIZ, "realtime", "web")

app = FastAPI(title="Labios Argentos — captura para adaptación")


def limpiar(texto: str) -> str:
    """Misma normalización que el dataset (descargar_procesar.limpiar)."""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = unidecode(texto.replace("ñ", "ENIE")).replace("ENIE", "ñ")
    return texto.strip()


def cargar_frases():
    """Las líneas numeradas ('N. frase') del markdown, en orden."""
    frases = []
    with open(_FRASES_MD, encoding="utf-8") as f:
        for linea in f:
            m = re.match(r"^(\d+)\.\s+(.+)$", linea.strip())
            if m:
                frases.append(m.group(2).strip())
    return frases


@app.get("/")
def index():
    return FileResponse(os.path.join(_WEB, "grabar.html"))


@app.get("/frases")
def frases():
    lista = cargar_frases()
    hechas = sorted(
        int(m.group(1))
        for n in (os.listdir(_GRAB) if os.path.isdir(_GRAB) else [])
        if (m := re.match(r"frase_(\d+)\.mp4$", n))
    )
    return JSONResponse({"frases": lista, "hechas": hechas})


@app.post("/guardar")
def guardar(clip: UploadFile = File(...), indice: int = Form(...), texto: str = Form(...)):
    os.makedirs(_GRAB, exist_ok=True)
    sufijo = os.path.splitext(clip.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(clip.file.read())
        crudo = tmp.name
    destino = os.path.join(_GRAB, f"frase_{indice:03d}.mp4")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", crudo,
             "-r", "25", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", destino],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return JSONResponse({"ok": False, "error": r.stderr[-300:]})
    finally:
        os.unlink(crudo)
    with open(os.path.join(_GRAB, f"frase_{indice:03d}.txt"), "w", encoding="utf-8") as f:
        f.write(limpiar(texto) + "\n")
    return JSONResponse({"ok": True, "indice": indice})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8001")))
