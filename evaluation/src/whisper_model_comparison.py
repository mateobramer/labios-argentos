from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

from unidecode import unidecode


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}


def normalizar_texto(texto: str) -> str:
    """Normaliza texto para comparar palabras sin diferencias cosmeticas."""
    texto = texto.lower()
    texto = texto.replace("\u00f1", "ENIE")
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = unidecode(texto).replace("ENIE", "\u00f1")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def nombre_seguro(texto: str, max_chars: int = 80) -> str:
    texto = normalizar_texto(texto)
    texto = texto.replace("\u00f1", "n")
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    return texto[:max_chars] or "video"


def buscar_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "No se encontro ffmpeg. Instalalo o agrega imageio-ffmpeg al entorno."
        ) from exc


def _yt_dlp_cmd() -> list[str]:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]

    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except ImportError as exc:
        raise RuntimeError("No se encontro yt-dlp en el entorno.") from exc


def descargar_video_yt(
    url: str,
    output_dir: str | Path,
    nombre_base: str = "azzaro_comparacion",
    max_height: int = 720,
    cookies_from_browser: str | None = None,
) -> Path:
    """Descarga un video de YouTube para el experimento y reutiliza si ya existe."""
    if not url or "PEGAR_URL" in url:
        raise ValueError("Completar VIDEO_URL con una URL real de YouTube.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existentes = [
        path
        for path in sorted(output_dir.glob(f"{nombre_base}.*"))
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if existentes:
        return existentes[0]

    cmd = _yt_dlp_cmd()
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    cmd += [
        "--no-part",
        "-f",
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
        "-o",
        str(output_dir / f"{nombre_base}.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)

    descargados = [
        path
        for path in sorted(output_dir.glob(f"{nombre_base}.*"))
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not descargados:
        raise FileNotFoundError(f"No se encontro video descargado en {output_dir}")
    return descargados[0]


def transcribir_whisper(
    video_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    language: str = "es",
    force: bool = False,
) -> dict[str, Any]:
    """Transcribe con Whisper y cachea el JSON para no repetir corridas caras."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / f"{nombre_seguro(video_path.stem)}__{model_name}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(video_path),
        language=language,
        verbose=False,
        fp16=False,
        word_timestamps=True,
    )

    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _palabras_segmento_fallback(segment: dict[str, Any]) -> list[dict[str, Any]]:
    texto = segment.get("text", "")
    tokens = [tok for tok in texto.split() if normalizar_texto(tok)]
    if not tokens:
        return []

    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    paso = max((end - start) / len(tokens), 0.0)
    palabras = []
    for idx, token in enumerate(tokens):
        palabras.append(
            {
                "word": token,
                "start": start + idx * paso,
                "end": start + (idx + 1) * paso,
            }
        )
    return palabras


def extraer_palabras(resultado: dict[str, Any], modelo: str) -> list[dict[str, Any]]:
    """Convierte segmentos Whisper a una lista plana de palabras con timestamps."""
    palabras: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(resultado.get("segments", [])):
        raw_words = segment.get("words") or _palabras_segmento_fallback(segment)
        for word_id, word in enumerate(raw_words):
            raw = str(word.get("word", "")).strip()
            norm = normalizar_texto(raw)
            if not norm:
                continue
            palabras.append(
                {
                    "model": modelo,
                    "segment_id": segment_id,
                    "word_id": word_id,
                    "word": raw,
                    "word_norm": norm,
                    "start": float(word.get("start", segment.get("start", 0.0))),
                    "end": float(word.get("end", segment.get("end", 0.0))),
                }
            )
    return palabras


def _unir(palabras: list[dict[str, Any]], key: str) -> str:
    return " ".join(str(p[key]).strip() for p in palabras if str(p[key]).strip())


def _rango_tiempo(*grupos: list[dict[str, Any]]) -> tuple[float, float]:
    tiempos = [p for grupo in grupos for p in grupo if p.get("start") is not None]
    if not tiempos:
        return 0.0, 0.0
    start = min(float(p["start"]) for p in tiempos)
    end = max(float(p["end"]) for p in tiempos)
    return start, end


def _contexto(
    palabras: list[dict[str, Any]],
    inicio: int,
    fin: int,
    ventana: int,
) -> str:
    left = max(0, inicio - ventana)
    right = min(len(palabras), fin + ventana)
    return _unir(palabras[left:right], "word")


def alinear_diferencias(
    palabras_turbo: list[dict[str, Any]],
    palabras_small: list[dict[str, Any]],
    contexto: int = 5,
) -> list[dict[str, Any]]:
    """Alinea palabras normalizadas y devuelve los grupos donde difieren."""
    seq_turbo = [p["word_norm"] for p in palabras_turbo]
    seq_small = [p["word_norm"] for p in palabras_small]
    matcher = SequenceMatcher(None, seq_turbo, seq_small, autojunk=False)

    diferencias: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        turbo_slice = palabras_turbo[i1:i2]
        small_slice = palabras_small[j1:j2]
        start, end = _rango_tiempo(turbo_slice, small_slice)
        diff_id = len(diferencias) + 1
        diferencias.append(
            {
                "diff_id": diff_id,
                "tipo": tag,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(max(end - start, 0.0), 3),
                "turbo_text": _unir(turbo_slice, "word"),
                "small_text": _unir(small_slice, "word"),
                "turbo_norm": _unir(turbo_slice, "word_norm"),
                "small_norm": _unir(small_slice, "word_norm"),
                "n_palabras_turbo": len(turbo_slice),
                "n_palabras_small": len(small_slice),
                "contexto_turbo": _contexto(palabras_turbo, i1, i2, contexto),
                "contexto_small": _contexto(palabras_small, j1, j2, contexto),
            }
        )

    return diferencias


def _unir_intervalos(
    intervalos: list[tuple[float, float]],
    gap: float,
) -> list[tuple[float, float]]:
    """Une intervalos solapados o separados por una pausa corta."""
    ordenados = sorted(
        (float(start), float(end))
        for start, end in intervalos
        if float(end) > float(start)
    )
    unidos: list[list[float]] = []
    for start, end in ordenados:
        if not unidos or start > unidos[-1][1] + gap:
            unidos.append([start, end])
        else:
            unidos[-1][1] = max(unidos[-1][1], end)
    return [(start, end) for start, end in unidos]


def _palabras_con_contexto(
    palabras: list[dict[str, Any]],
    start: float,
    end: float,
    contexto: int,
) -> list[dict[str, Any]]:
    """Toma palabras del intervalo y agrega vecinos completos como contexto."""
    indices = [
        idx
        for idx, palabra in enumerate(palabras)
        if float(palabra["end"]) >= start and float(palabra["start"]) <= end
    ]
    if not indices:
        return []

    left = max(indices[0] - contexto, 0)
    right = min(indices[-1] + contexto + 1, len(palabras))
    return palabras[left:right]


def _palabras_en_intervalo(
    palabras: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Selecciona palabras cuyo centro temporal cae dentro del clip."""
    seleccionadas = []
    for palabra in palabras:
        p_start = float(palabra["start"])
        p_end = float(palabra["end"])
        centro = (p_start + p_end) / 2
        if start <= centro <= end:
            seleccionadas.append(palabra)
    return seleccionadas


def construir_casos_tres_modelos(
    palabras_por_modelo: dict[str, list[dict[str, Any]]],
    modelos: tuple[str, str, str] = ("turbo", "small", "large"),
    gap_diferencias: float = 0.55,
    contexto_palabras: int = 1,
) -> list[dict[str, Any]]:
    """Construye casos alineados donde los tres modelos producen textos distintos.

    Los intervalos iniciales salen de las diferencias palabra-a-palabra de cada par.
    Luego se expanden a palabras completas y se usa exactamente el mismo rango temporal
    para el clip y para recuperar el texto de los tres modelos.
    """
    intervalos = []
    for modelo_a, modelo_b in combinations(modelos, 2):
        diferencias = alinear_diferencias(
            palabras_por_modelo[modelo_a],
            palabras_por_modelo[modelo_b],
            contexto=0,
        )
        intervalos.extend(
            (float(diferencia["start"]), float(diferencia["end"]))
            for diferencia in diferencias
        )

    intervalos_base = _unir_intervalos(intervalos, gap=gap_diferencias)
    intervalos_expandidos = []
    for start, end in intervalos_base:
        palabras_contexto = [
            palabra
            for modelo in modelos
            for palabra in _palabras_con_contexto(
                palabras_por_modelo[modelo],
                start,
                end,
                contexto_palabras,
            )
        ]
        if not palabras_contexto:
            continue
        intervalos_expandidos.append(
            (
                min(float(palabra["start"]) for palabra in palabras_contexto),
                max(float(palabra["end"]) for palabra in palabras_contexto),
            )
        )

    # El contexto puede hacer que dos diferencias vecinas se solapen.
    intervalos_finales = _unir_intervalos(intervalos_expandidos, gap=0.05)

    casos = []
    for start, end in intervalos_finales:
        textos = {}
        textos_norm = {}
        for modelo in modelos:
            seleccionadas = _palabras_en_intervalo(
                palabras_por_modelo[modelo],
                start,
                end,
            )
            textos[modelo] = _unir(seleccionadas, "word")
            textos_norm[modelo] = normalizar_texto(textos[modelo])

        if not all(textos_norm.values()):
            continue
        if len(set(textos_norm.values())) != len(modelos):
            continue

        casos.append(
            {
                "diff_id": len(casos) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                **{
                    f"transcripcion_{modelo}": textos[modelo]
                    for modelo in modelos
                },
            }
        )

    return casos


def resumen_diferencias(
    diferencias: list[dict[str, Any]],
    palabras_turbo: list[dict[str, Any]],
    palabras_small: list[dict[str, Any]],
) -> dict[str, Any]:
    total_turbo = len(palabras_turbo)
    total_small = len(palabras_small)
    palabras_turbo_diff = sum(d["n_palabras_turbo"] for d in diferencias)
    palabras_small_diff = sum(d["n_palabras_small"] for d in diferencias)

    base = max(total_turbo, 1)
    return {
        "total_palabras_turbo": total_turbo,
        "total_palabras_small": total_small,
        "grupos_con_diferencias": len(diferencias),
        "palabras_turbo_en_diferencias": palabras_turbo_diff,
        "palabras_small_en_diferencias": palabras_small_diff,
        "diferencias_por_1000_palabras_turbo": round(
            palabras_turbo_diff * 1000 / base, 2
        ),
    }


def exportar_clips_diferencias(
    video_path: str | Path,
    diferencias: list[dict[str, Any]],
    output_dir: str | Path,
    margen: float = 1.25,
    max_clips: int = 40,
) -> list[dict[str, Any]]:
    """Exporta clips alrededor de cada diferencia para revisar quien tiene razon."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = buscar_ffmpeg()

    exportadas: list[dict[str, Any]] = []
    for diferencia in diferencias[:max_clips]:
        start = max(float(diferencia["start"]) - margen, 0.0)
        end = max(float(diferencia["end"]) + margen, start + 0.5)
        duration = end - start
        clip_path = output_dir / f"diff_{int(diferencia['diff_id']):03d}.mp4"

        if not clip_path.exists():
            cmd = [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-preset",
                "veryfast",
                str(clip_path),
            ]
            subprocess.run(cmd, check=True)

        fila = dict(diferencia)
        fila["clip_start"] = round(start, 3)
        fila["clip_end"] = round(end, 3)
        fila["clip_path"] = str(clip_path)
        exportadas.append(fila)

    return exportadas


def exportar_clips_revision_webm(
    video_path: str | Path,
    diferencias: list[dict[str, Any]],
    output_dir: str | Path,
    margen: float = 0.08,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Exporta clips WebM compatibles con el renderer de notebooks de VSCode.

    VSCode distribuye VP8 y Vorbis, mientras que AAC no esta incluido. Por eso estos
    clips de revision usan WebM aunque el dataset principal siga guardando MP4.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = buscar_ffmpeg()

    exportadas = []
    for diferencia in diferencias:
        start = max(float(diferencia["start"]) - margen, 0.0)
        end = max(float(diferencia["end"]) + margen, start + 0.5)
        duration = end - start
        clip_path = output_dir / f"diff_{int(diferencia['diff_id']):03d}.webm"

        if force or not clip_path.exists():
            cmd = [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "libvpx",
                "-crf",
                "18",
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "libvorbis",
                "-q:a",
                "5",
                str(clip_path),
            ]
            subprocess.run(cmd, check=True)

        fila = dict(diferencia)
        fila["clip_start"] = round(start, 3)
        fila["clip_end"] = round(end, 3)
        fila["clip_path"] = str(clip_path.resolve())
        exportadas.append(fila)

    return exportadas

