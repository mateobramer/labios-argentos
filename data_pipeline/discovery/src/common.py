"""Utilidades compartidas para data discovery.

El modulo evita descargar datos pesados por defecto. Los scripts escriben CSV/JSON
livianos y dejan videos/audios temporales bajo rutas ignoradas por git.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


RAIZ_REPO = Path(__file__).resolve().parents[2]
DATA_DISCOVERY = RAIZ_REPO / "data_pipeline/discovery"
OUTPUTS = DATA_DISCOVERY / "outputs"
SAMPLE_METADATA = OUTPUTS / "sample_metadata"
CONTACT_SHEETS = OUTPUTS / "contact_sheets"
SAMPLES = OUTPUTS / "samples"

FUENTE_CSV = DATA_DISCOVERY / "sources_seed.csv"
CANDIDATE_VIDEOS_CSV = OUTPUTS / "candidate_videos.csv"
CANDIDATE_SCORES_CSV = OUTPUTS / "candidate_scores.csv"
SHORTLIST_CSV = OUTPUTS / "shortlist_recommended.csv"
REJECTED_CSV = OUTPUTS / "rejected_candidates.csv"
DIVERSITY_CSV = OUTPUTS / "source_diversity_report.csv"
TARGET_PROGRESS_MD = OUTPUTS / "target_progress.md"
REVIEW_INDEX_MD = OUTPUTS / "review_index.md"
INGEST_PLAN_MD = OUTPUTS / "ingest_plan_v1.md"


SOURCE_SEED_FIELDS = [
    "source_url",
    "title",
    "channel",
    "source_type",
    "expected_accent",
    "expected_speaker_count",
    "notes",
]

CANDIDATE_FIELDS = [
    "url",
    "title",
    "channel",
    "channel_url",
    "video_id",
    "duration",
    "duration_minutes",
    "width",
    "height",
    "fps",
    "upload_date",
    "view_count",
    "language",
    "tags",
    "source_type",
    "expected_accent",
    "expected_speaker_count",
    "query",
    "search_rank",
    "notes",
]

SCORE_FIELDS = [
    "url",
    "title",
    "channel",
    "video_id",
    "duration_minutes",
    "width",
    "height",
    "fps",
    "source_type",
    "expected_accent",
    "visual_quality_score",
    "audio_quality_score",
    "context_score",
    "total_score",
    "visual_decision",
    "audio_decision",
    "decision",
    "reasons",
    "recommended_use",
    "usable_minutes_estimate",
    "accepted_clips_estimate",
    "uncertainty",
]

SHORTLIST_FIELDS = [
    "rank",
    "url",
    "title",
    "channel",
    "decision",
    "total_score",
    "visual_quality_score",
    "audio_quality_score",
    "context_score",
    "usable_minutes_estimate",
    "accepted_clips_estimate",
    "recommended_use",
    "reasons",
    "contact_sheet_path",
    "notes",
]

REJECTED_FIELDS = [
    "url",
    "title",
    "channel",
    "decision",
    "total_score",
    "visual_quality_score",
    "audio_quality_score",
    "context_score",
    "reject_reasons",
]

DIVERSITY_FIELDS = [
    "channel",
    "source_type",
    "accepted_videos",
    "usable_minutes_estimate",
    "accepted_clips_estimate",
    "avg_total_score",
    "avg_visual_score",
    "avg_audio_score",
    "decision",
    "notes",
]


def asegurar_directorios() -> None:
    for path in [DATA_DISCOVERY, OUTPUTS, SAMPLE_METADATA, CONTACT_SHEETS, SAMPLES]:
        path.mkdir(parents=True, exist_ok=True)


def configurar_salida_utf8() -> None:
    """Evita fallas de impresion por consola Windows cp1252."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def buscar_ejecutable(nombre: str) -> str | None:
    exe = shutil.which(nombre)
    if exe:
        return exe
    scripts_python = Path(sys.executable).parent / "Scripts"
    candidato = scripts_python / (nombre + (".exe" if os.name == "nt" else ""))
    if candidato.exists():
        return str(candidato)
    return None


def comando_yt_dlp() -> list[str]:
    exe = buscar_ejecutable("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depende del entorno local
        raise RuntimeError("No se encontro yt-dlp. Instalar requirements.txt.") from exc
    return [sys.executable, "-m", "yt_dlp"]


def comando_ffmpeg() -> str:
    exe = buscar_ejecutable("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:  # pragma: no cover - depende del entorno local
        raise RuntimeError("No se encontro ffmpeg ni imageio-ffmpeg.") from exc


def correr(cmd: list[str], *, timeout: int = 120, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def leer_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: limpiar_valor_csv(row.get(field, "")) for field in fieldnames})


def limpiar_valor_csv(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(x) for x in value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 4)
    if value is None:
        return ""
    return value


def leer_jsonl(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def cargar_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def guardar_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def slug(texto: str, max_len: int = 80) -> str:
    texto = texto.lower()
    texto = re.sub(r"https?://", "", texto)
    texto = re.sub(r"[^a-z0-9_-]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto[:max_len] or "sin_id"


def texto_normalizado(*partes: Any) -> str:
    return " ".join(str(p or "").lower() for p in partes)


def clamp(valor: float, bajo: float = 0.0, alto: float = 1.0) -> float:
    return max(bajo, min(alto, valor))


def score_rango(valor: float, malo: float, bueno: float) -> float:
    if bueno <= malo:
        return 0.0
    return clamp((valor - malo) / (bueno - malo))


def score_intervalo(valor: float, bajo_malo: float, bajo_bueno: float, alto_bueno: float, alto_malo: float) -> float:
    if valor < bajo_bueno:
        return score_rango(valor, bajo_malo, bajo_bueno)
    if valor <= alto_bueno:
        return 1.0
    if valor >= alto_malo:
        return 0.0
    return clamp((alto_malo - valor) / (alto_malo - alto_bueno))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def decision_orden(decision: str) -> int:
    return {
        "strong_accept": 0,
        "accept": 1,
        "maybe_review": 2,
        "reject": 3,
        "error": 4,
    }.get(decision, 5)


def formatear_minutos(minutos: float) -> str:
    return f"{minutos:.1f}"
