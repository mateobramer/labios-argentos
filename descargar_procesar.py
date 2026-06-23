import os
import sys
import shutil
import subprocess
import json
import re
import tempfile
import whisper
from unidecode import unidecode

DATA_DIR = "data"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
MODELO_WHISPER = None
MODELO_WHISPER_NOMBRE = os.environ.get("WHISPER_MODEL", "turbo")

# ffmpeg y yt-dlp deben estar en el PATH del sistema (ver README).
# Si yt-dlp fue instalado con pip en Windows, suele quedar en Scripts.
scripts_python = os.path.join(os.path.dirname(sys.executable), "Scripts")
if os.path.isdir(scripts_python):
    os.environ["PATH"] += os.pathsep + scripts_python

# En Windows, si ffmpeg se instalo con winget pero no quedo en el PATH,
# agregamos su carpeta tipica solo si existe.
if os.name == "nt" and shutil.which("ffmpeg") is None:
    posibles = [
        os.path.expandvars(
            r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.1.1-essentials_build\bin"
        ),
    ]
    for ruta in posibles:
        if os.path.isdir(ruta):
            os.environ["PATH"] += os.pathsep + ruta
            break

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
if FFMPEG is not None:
    os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG)
    if os.name == "nt" and os.path.basename(FFMPEG).lower() != "ffmpeg.exe":
        ffmpeg_dir = os.path.join(tempfile.gettempdir(), "labios-argentos-bin")
        os.makedirs(ffmpeg_dir, exist_ok=True)
        ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_exe):
            shutil.copy2(FFMPEG, ffmpeg_exe)
        FFMPEG = ffmpeg_exe
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

YTDLP = shutil.which("yt-dlp")
if YTDLP is None:
    try:
        import yt_dlp  # noqa: F401
        YTDLP = [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass

# Verificacion temprana: avisamos claro si faltan las herramientas externas.
if FFMPEG is None:
    print("ERROR: no se encontro 'ffmpeg'. Ver instrucciones en el README.")
    sys.exit(1)
if YTDLP is None:
    print("ERROR: no se encontro 'yt-dlp'. Ver instrucciones en el README.")
    sys.exit(1)

if isinstance(YTDLP, str):
    YTDLP_CMD = [YTDLP]
else:
    YTDLP_CMD = YTDLP

def cargar_modelo():
    global MODELO_WHISPER
    if MODELO_WHISPER is None:
        print(f"Cargando modelo Whisper {MODELO_WHISPER_NOMBRE}...")
        MODELO_WHISPER = whisper.load_model(MODELO_WHISPER_NOMBRE)
    return MODELO_WHISPER

def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = unidecode(texto.replace("ñ", "ENIE")).replace("ENIE", "ñ")
    return texto.strip()

def nombre_carpeta(titulo):
    # limpia el titulo para usarlo como nombre de carpeta
    titulo = re.sub(r'[<>:"/\\|?*]', '', titulo)
    titulo = titulo[:50].strip()
    return titulo

def bajar_video(url):
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    # primero obtenemos el titulo sin bajar
    cmd_titulo = YTDLP_CMD + ["--print", "title", url]
    titulo = subprocess.check_output(cmd_titulo).decode("utf-8", errors="ignore").strip()
    carpeta = nombre_carpeta(titulo)
    
    # carpeta del video
    video_dir = os.path.join(VIDEOS_DIR, carpeta)
    os.makedirs(video_dir, exist_ok=True)

    archivos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if archivos:
        video_path = os.path.join(video_dir, archivos[0])
        print(f"Usando video existente: {video_path}")
        return video_path, carpeta
    
    # bajar el video
    cmd = YTDLP_CMD + [
        "--no-part",
        "-o", os.path.join(video_dir, "%(title)s.%(ext)s"),
        "--format", "mp4",
        url
    ]
    subprocess.run(cmd, check=True)
    
    # encontrar el mp4 descargado
    archivos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not archivos:
        raise FileNotFoundError(f"No se encontro ningun .mp4 descargado en {video_dir}")
    video_path = os.path.join(video_dir, archivos[0])
    
    return video_path, carpeta

def transcribir(video_path, carpeta):
    corpus_dir = os.path.join(CORPUS_DIR, carpeta)
    json_path = os.path.join(corpus_dir, "transcripcion.json")
    os.makedirs(corpus_dir, exist_ok=True)

    if os.path.exists(json_path):
        print("Cargando transcripcion existente...")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Transcribiendo con Whisper...")
    modelo = cargar_modelo()
    # word_timestamps=True es CLAVE: sin esto Whisper solo da tiempos a nivel
    # segmento, que derivan en habla continua y hacen que el corte caiga corrido
    # respecto del texto (el clip muestra/dice la frase vecina). Con tiempos por
    # palabra cortamos en pausas reales y el rango del clip calza con su texto.
    resultado = modelo.transcribe(
        video_path, language="es", verbose=False, fp16=False, word_timestamps=True
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False)
    print(f"JSON guardado: {json_path}")

    return resultado

def guardar_corpus(resultado, carpeta):
    txt_path = os.path.join(CORPUS_DIR, carpeta, "corpus.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in resultado["segments"]:
            texto = limpiar(seg["text"])
            if len(texto.split()) >= 3:
                f.write(texto + "\n")
    print(f"Corpus guardado: {txt_path}")

# Parametros de corte. Un clip se cierra cuando ya dura DUR_MIN y aparece una
# pausa real (silencio entre palabras >= GAP_CORTE); si nunca hay pausa, se fuerza
# el corte al llegar a DUR_MAX. PAD agrega un margen chico para no comerse el
# primer/ultimo fonema.
DUR_MIN = 3.0
DUR_MAX = 10.0
GAP_CORTE = 0.40
PAD = 0.08


def extraer_palabras(resultado):
    """Aplana las palabras con timestamps de todos los segmentos.

    Requiere haber transcripto con word_timestamps=True. Si el resultado no trae
    'words' (p. ej. una transcripcion.json vieja), devuelve None y el corte cae al
    fallback por segmentos.
    """
    palabras = []
    for seg in resultado.get("segments", []):
        words = seg.get("words")
        if not words:
            return None
        for w in words:
            texto = str(w.get("word", "")).strip()
            if texto and w.get("start") is not None and w.get("end") is not None:
                palabras.append({"word": texto, "start": float(w["start"]), "end": float(w["end"])})
    return palabras or None


def agrupar_palabras(palabras):
    """Agrupa palabras en clips cortando en pausas reales.

    Cada grupo abarca [primera.start, ultima.end] en tiempo de reloj real. Se cierra
    cuando ya paso DUR_MIN y hay un silencio >= GAP_CORTE, o cuando se llega a DUR_MAX
    (corte forzado). El rango del clip y su texto salen de los MISMOS tiempos de
    palabra, asi que no hay deriva clip<->texto.
    """
    grupos = []
    actual = []
    for w in palabras:
        if actual:
            span = actual[-1]["end"] - actual[0]["start"]
            gap = w["start"] - actual[-1]["end"]
            if (span >= DUR_MIN and gap >= GAP_CORTE) or span >= DUR_MAX:
                grupos.append(actual)
                actual = []
        actual.append(w)
    if actual:
        if (actual[-1]["end"] - actual[0]["start"]) >= 1.0 or not grupos:
            grupos.append(actual)
        else:
            grupos[-1].extend(actual)  # coda muy corta -> al grupo anterior
    return [(g[0]["start"], g[-1]["end"], " ".join(x["word"] for x in g)) for g in grupos]


def agrupar_segmentos(resultado):
    """Fallback (transcripcion sin word-timestamps): agrupa por segmentos.

    Menos preciso que agrupar_palabras y propenso a deriva en habla continua; solo
    se usa con caches viejos. Devuelve [(inicio, fin, texto), ...].
    """
    grupos, actual, dur = [], [], 0.0
    for seg in resultado["segments"]:
        actual.append(seg)
        dur += seg["end"] - seg["start"]
        if dur >= DUR_MIN:
            grupos.append(actual); actual, dur = [], 0.0
    if actual and dur >= 2.0:
        grupos.append(actual)
    return [(g[0]["start"], g[-1]["end"], " ".join(s["text"].strip() for s in g)) for g in grupos]


def cortar_clips(video_path, resultado, carpeta):
    clips_dir = os.path.join(CLIPS_DIR, carpeta)
    os.makedirs(clips_dir, exist_ok=True)

    palabras = extraer_palabras(resultado)
    if palabras is not None:
        grupos = agrupar_palabras(palabras)
    else:
        print("  (aviso: la transcripcion no tiene word-timestamps; corte por segmentos, menos preciso)")
        grupos = agrupar_segmentos(resultado)

    clips = []
    for i, (inicio, fin, texto) in enumerate(grupos):
        inicio_c = max(0.0, inicio - PAD)
        duracion = (fin + PAD) - inicio_c

        nombre = f"clip_{i:04d}"
        clip_path = os.path.join(clips_dir, f"{nombre}.mp4")
        txt_path = os.path.join(clips_dir, f"{nombre}.txt")

        cmd = [
            FFMPEG,
            "-nostdin",
            "-loglevel", "error",
            "-y",
            "-ss", str(inicio_c),
            "-i", video_path,
            "-t", str(duracion),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "veryfast",
            clip_path
        ]
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(limpiar(texto))

        clips.append(nombre)
        print(f"Clip {len(clips):04d}: '{texto[:50]}' ({duracion:.1f}s)")

    print(f"\nTotal clips: {len(clips)}")
    print(f"Guardados en: {clips_dir}/")
    return clips

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python descargar_procesar.py URL_YOUTUBE [URL_YOUTUBE ...]")
        sys.exit(1)

    urls = sys.argv[1:]

    for url in urls:
        print("=" * 80)
        print(f"Bajando video...")
        video_path, carpeta = bajar_video(url)
        print(f"Video guardado: {video_path}")
        print(f"Carpeta: {carpeta}")

        resultado = transcribir(video_path, carpeta)
        guardar_corpus(resultado, carpeta)
        cortar_clips(video_path, resultado, carpeta)
