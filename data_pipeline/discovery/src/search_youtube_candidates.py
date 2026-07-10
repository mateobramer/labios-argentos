"""Busca candidatos de YouTube sin descargar videos.

Salida principal:
- data_pipeline/discovery/outputs/candidate_videos.csv
- data_pipeline/discovery/sources_seed.csv
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Any

from data_pipeline.discovery.src.common import (
    CANDIDATE_FIELDS,
    CANDIDATE_VIDEOS_CSV,
    FUENTE_CSV,
    SOURCE_SEED_FIELDS,
    asegurar_directorios,
    comando_yt_dlp,
    configurar_salida_utf8,
    correr,
    escribir_csv,
    leer_csv,
    leer_jsonl,
    texto_normalizado,
)


DEFAULT_QUERIES = [
    "podcast argentino entrevista camara fija",
    "streamer argentino charla larga facecam",
    "entrevista argentina completa",
    "charla argentina YouTube cara visible",
    "youtuber argentino hablando a camara",
    "podcast rioplatense video completo",
    "entrevista Coscu streamers argentinos",
    "influencers argentinos entrevista completa",
    "canales argentinos conversacion larga",
    "clases charlas argentinas speaker visible",
    "podcast argentino video completo invitado",
    "monologo argentino YouTube",
]

PALABRAS_ARG = [
    "argentina",
    "argentino",
    "argentina",
    "buenos aires",
    "caba",
    "rioplatense",
    "mate",
    "che",
    "voseo",
    "coscu",
    "migue granados",
    "olga",
    "luzu",
    "urbana play",
    "gelatina",
]

PALABRAS_RECHAZO = [
    "shorts",
    "#shorts",
    "short ",
    "reel",
    "tiktok",
    "compilado",
    "compilacion",
    "musica",
    "cancion",
    "karaoke",
    "letra",
    "doblaje",
    "trailer",
    "clip oficial",
]


def inferir_source_type(title: str, channel: str, description: str = "") -> str:
    texto = texto_normalizado(title, channel, description)
    if "podcast" in texto:
        return "podcast"
    if any(k in texto for k in ["entrevista", "mano a mano", "completa"]):
        return "interview"
    if any(k in texto for k in ["stream", "streamer", "twitch", "facecam"]):
        return "streamer"
    if any(k in texto for k in ["clase", "charla", "conferencia", "educacion", "curso"]):
        return "educational"
    if any(k in texto for k in ["vlog", "storytime", "hablando a camara"]):
        return "vlog"
    if any(k in texto for k in ["monologo", "reflexion", "opinion"]):
        return "monologue"
    if any(k in texto for k in ["conversacion", "debate"]):
        return "conversation"
    return "other"


def inferir_acento(title: str, channel: str, description: str = "", tags: Any = None) -> str:
    texto = texto_normalizado(title, channel, description, " ".join(tags or []))
    if any(k in texto for k in PALABRAS_ARG):
        return "argentino/rioplatense_probable"
    return "unknown"


def inferir_speaker_count(source_type: str, title: str) -> str:
    texto = texto_normalizado(title)
    if source_type in {"monologue", "educational", "vlog", "streamer"}:
        return "1"
    if any(k in texto for k in ["panel", "mesa", "debate"]):
        return "3+"
    if source_type in {"podcast", "interview", "conversation"}:
        return "1-2"
    return "unknown"


def es_candidato_descartable(info: dict[str, Any], min_duration: int, max_duration: int) -> tuple[bool, str]:
    title = str(info.get("title") or "")
    description = str(info.get("description") or "")
    duration = int(info.get("duration") or 0)
    texto = texto_normalizado(title, description)
    if duration and duration < min_duration:
        return True, f"duracion_menor_a_{min_duration}s"
    if max_duration and duration > max_duration:
        return True, f"duracion_mayor_a_{max_duration}s"
    for palabra in PALABRAS_RECHAZO:
        if palabra in texto:
            return True, f"rechazo_keyword:{palabra}"
    return False, ""


def normalizar_info(info: dict[str, Any], query: str, rank: int) -> dict[str, Any]:
    title = str(info.get("title") or "")
    channel = str(info.get("channel") or info.get("uploader") or "")
    description = str(info.get("description") or "")
    tags = info.get("tags") or []
    source_type = inferir_source_type(title, channel, description)
    duration = int(info.get("duration") or 0)
    url = info.get("webpage_url") or info.get("url") or ""
    video_id = info.get("id") or info.get("display_id") or ""
    if video_id and "youtube.com" not in str(url) and "youtu.be" not in str(url):
        url = f"https://www.youtube.com/watch?v={video_id}"
    return {
        "url": url,
        "title": title,
        "channel": channel,
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "video_id": video_id,
        "duration": duration,
        "duration_minutes": round(duration / 60, 3) if duration else "",
        "width": info.get("width") or "",
        "height": info.get("height") or "",
        "fps": info.get("fps") or "",
        "upload_date": info.get("upload_date") or "",
        "view_count": info.get("view_count") or "",
        "language": info.get("language") or "",
        "tags": "|".join(str(t) for t in tags[:20]),
        "source_type": source_type,
        "expected_accent": inferir_acento(title, channel, description, tags),
        "expected_speaker_count": inferir_speaker_count(source_type, title),
        "query": query,
        "search_rank": rank,
        "notes": "ytsearch_metadata_no_download",
    }


def buscar_query(query: str, max_results: int, min_duration: int, max_duration: int) -> list[dict[str, Any]]:
    cmd = comando_yt_dlp() + [
        "--dump-json",
        "--skip-download",
        "--no-playlist",
        f"ytsearch{max_results}:{query}",
    ]
    result = correr(cmd, timeout=180)
    if result.returncode != 0:
        return [
            {
                "url": "",
                "title": "",
                "channel": "",
                "video_id": "",
                "source_type": "other",
                "expected_accent": "unknown",
                "expected_speaker_count": "unknown",
                "query": query,
                "search_rank": "",
                "notes": "search_error:" + (result.stderr or result.stdout).strip()[:240],
            }
        ]

    rows = []
    for rank, info in enumerate(leer_jsonl(result.stdout), start=1):
        descartable, reason = es_candidato_descartable(info, min_duration, max_duration)
        row = normalizar_info(info, query, rank)
        if descartable:
            row["notes"] = f"prefilter_reject:{reason}"
        rows.append(row)
    return rows


def construir_sources_seed(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    por_fuente: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in candidates:
        if str(row.get("notes", "")).startswith("prefilter_reject"):
            continue
        channel = str(row.get("channel") or "unknown")
        key = row.get("channel_url") or channel
        if key not in por_fuente:
            por_fuente[key] = {
                "source_url": row.get("channel_url") or row.get("url") or "",
                "title": row.get("title") or "",
                "channel": channel,
                "source_type": row.get("source_type") or "other",
                "expected_accent": row.get("expected_accent") or "unknown",
                "expected_speaker_count": row.get("expected_speaker_count") or "unknown",
                "notes": "seed_from_youtube_search",
            }
    return list(por_fuente.values())


def deduplicar(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        key = row.get("video_id") or row.get("url") or f"{row.get('query')}:{row.get('search_rank')}"
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())


def main() -> int:
    configurar_salida_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", dest="queries", help="Query de busqueda. Repetible.")
    parser.add_argument("--max-results", type=int, default=8, help="Resultados por query.")
    parser.add_argument("--min-duration", type=int, default=8 * 60, help="Duracion minima en segundos.")
    parser.add_argument("--max-duration", type=int, default=4 * 60 * 60, help="Duracion maxima en segundos.")
    parser.add_argument("--append", action="store_true", help="Mantener candidatos existentes y sumar nuevos.")
    args = parser.parse_args()

    asegurar_directorios()
    queries = args.queries or DEFAULT_QUERIES
    rows = leer_csv(CANDIDATE_VIDEOS_CSV) if args.append else []
    for query in queries:
        rows.extend(buscar_query(query, args.max_results, args.min_duration, args.max_duration))

    rows = deduplicar(rows)
    escribir_csv(CANDIDATE_VIDEOS_CSV, rows, CANDIDATE_FIELDS)
    escribir_csv(FUENTE_CSV, construir_sources_seed(rows), SOURCE_SEED_FIELDS)
    print(f"candidatos={len(rows)} -> {CANDIDATE_VIDEOS_CSV}")
    print(f"fuentes_seed={len(construir_sources_seed(rows))} -> {FUENTE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
