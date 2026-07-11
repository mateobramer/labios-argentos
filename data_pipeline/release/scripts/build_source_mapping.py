"""Mapea fuentes originales para Argentina existing.

El script usa evidencias livianas:

- manifest argentino existente;
- config del bucket fuente;
- outputs de data_pipeline/discovery;
- fallback `yt-dlp ytsearch` para titulos no resueltos.

No descarga videos completos.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_pipeline/release"
REPORTS_DIR = OUT_DIR / "reports"
METADATA_DIR = OUT_DIR / "source_metadata"

ARG_EXISTING = OUT_DIR / "argentina_existing_manifest.csv"
CANDIDATE_SCORES = ROOT / "data_pipeline/discovery" / "outputs" / "candidate_scores.csv"
SHORTLIST = ROOT / "data_pipeline/discovery" / "outputs" / "shortlist_recommended.csv"
SOURCE_MAPPING = OUT_DIR / "source_mapping.csv"
REPORT = REPORTS_DIR / "source_reconstruction_report.md"

SOURCE_BUCKET = "gs://labios-argentos-vsr-dataset"
CANDIDATOS_CSV = f"{SOURCE_BUCKET}/config/candidatos_v2_FINAL.csv"

FIELDS = [
    "source_id",
    "spk",
    "titulo",
    "clip_count",
    "example_clips",
    "n_frames_min",
    "n_frames_mean",
    "n_frames_max",
    "text_examples",
    "candidate_url",
    "candidate_video_id",
    "candidate_title",
    "candidate_channel",
    "candidate_duration",
    "match_method",
    "match_score",
    "match_confidence",
    "has_youtube_subs",
    "has_youtube_auto_subs",
    "source_status",
    "notes",
]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


GSUTIL = tool("gsutil")
YTDLP = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    return re.sub(r"[^a-z0-9ñ]+", " ", value).strip()


def similarity(a: str, b: str) -> float:
    na = norm(a)
    nb = norm(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    if na in nb or nb in na:
        ratio = max(ratio, min(len(na), len(nb)) / max(len(na), len(nb)))
    return round(ratio * 100, 2)


def source_id(spk: str, titulo: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{spk}__{titulo}").strip("_")
    return value[:120] or "unknown_source"


def gsutil_cat(path: str) -> str:
    result = run([GSUTIL, "cat", path], timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"No se pudo leer {path}: {result.stderr[-1000:]}")
    return result.stdout


def group_sources(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("spk", ""), row.get("titulo", ""))].append(row)

    sources: list[dict[str, object]] = []
    for (spk, titulo), items in sorted(grouped.items()):
        frame_values = [int(float(row.get("n_frames") or 0)) for row in items]
        text_examples = [row.get("text_large_existing", "") for row in items[:3]]
        example_clips = [row.get("clip", "") for row in items[:5]]
        urls = [row.get("url", "") for row in items if row.get("url")]
        sources.append(
            {
                "source_id": source_id(spk, titulo),
                "spk": spk,
                "titulo": titulo,
                "clip_count": len(items),
                "example_clips": "|".join(example_clips),
                "n_frames_min": min(frame_values) if frame_values else "",
                "n_frames_mean": round(mean(frame_values), 2) if frame_values else "",
                "n_frames_max": max(frame_values) if frame_values else "",
                "text_examples": " || ".join(text_examples),
                "manifest_url": urls[0] if urls else "",
            }
        )
    return sources


def load_config_candidates() -> list[dict[str, str]]:
    return read_csv_text(gsutil_cat(CANDIDATOS_CSV))


def load_discovery_candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in [CANDIDATE_SCORES, SHORTLIST]:
        if path.exists():
            rows.extend(read_csv(path))
    return rows


def split_spk_to_n(spk: str) -> str:
    match = re.fullmatch(r"f0*(\d+)", (spk or "").strip().casefold())
    return str(int(match.group(1))) if match else ""


def config_candidates_for(source: dict[str, object], config_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    spk = str(source["spk"])
    titulo = str(source["titulo"])
    n = split_spk_to_n(spk)
    candidates: list[dict[str, object]] = []
    for row in config_rows:
        url = row.get("url", "")
        if not url:
            continue
        method = ""
        score = 0.0
        if n and str(row.get("n", "")).strip() == n:
            method = "config_spk_n"
            # El `n` del config ayuda a proponer un candidato, pero no prueba que
            # el URL sea el video fuente exacto del titulo en splits. La confianza
            # final la decide la similitud contra metadata/ytsearch.
            score = 52.0
        else:
            score = similarity(titulo, row.get("hablante", ""))
            if score >= 45:
                method = "config_hablante_fuzzy"
        if method:
            candidates.append(
                {
                    "url": url,
                    "title": row.get("hablante", ""),
                    "channel": "",
                    "duration": "",
                    "method": method,
                    "score": score,
                    "raw": row,
                }
            )
    return candidates


def discovery_candidates_for(source: dict[str, object], discovery_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    titulo = str(source["titulo"])
    candidates: list[dict[str, object]] = []
    for row in discovery_rows:
        url = row.get("url", "")
        if not url:
            continue
        score = similarity(titulo, row.get("title", ""))
        if score >= 55:
            candidates.append(
                {
                    "url": url,
                    "title": row.get("title", ""),
                    "channel": row.get("channel", ""),
                    "duration": row.get("duration_minutes", ""),
                    "method": "data_pipeline/discovery_title_fuzzy",
                    "score": score,
                    "raw": row,
                }
            )
    return candidates


def ytdlp_cmd() -> list[str]:
    if YTDLP:
        return [YTDLP]
    return [sys.executable, "-m", "yt_dlp"]


def ytsearch_candidates(source: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
    query = str(source["titulo"])
    args = ytdlp_cmd() + [
        "--dump-json",
        "--skip-download",
        "--ignore-errors",
        "--no-warnings",
        "--playlist-items",
        f"1:{limit}",
        f"ytsearch{limit}:{query}",
    ]
    result = run(args, timeout=120)
    if result.returncode != 0 and not result.stdout.strip():
        return []
    candidates = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            meta = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = meta.get("title", "")
        score = similarity(query, title)
        candidates.append(
            {
                "url": meta.get("webpage_url") or meta.get("original_url") or meta.get("url", ""),
                "title": title,
                "channel": meta.get("channel") or meta.get("uploader", ""),
                "duration": meta.get("duration") or "",
                "method": "ytsearch_title",
                "score": score,
                "raw": meta,
            }
        )
    return candidates


def fetch_metadata(url: str, source_id_value: str) -> dict[str, object]:
    if not url:
        return {}
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    args = ytdlp_cmd() + [
        "--dump-json",
        "--skip-download",
        "--ignore-errors",
        "--no-warnings",
        url,
    ]
    result = run(args, timeout=120)
    if result.returncode != 0 and not result.stdout.strip():
        return {}
    first_line = next((line for line in result.stdout.splitlines() if line.strip().startswith("{")), "")
    if not first_line:
        return {}
    try:
        meta = json.loads(first_line)
    except json.JSONDecodeError:
        return {}
    slim = {
        "id": meta.get("id"),
        "webpage_url": meta.get("webpage_url") or url,
        "title": meta.get("title"),
        "channel": meta.get("channel") or meta.get("uploader"),
        "duration": meta.get("duration"),
        "description": meta.get("description"),
        "upload_date": meta.get("upload_date"),
        "subtitles": sorted((meta.get("subtitles") or {}).keys()),
        "automatic_captions": sorted((meta.get("automatic_captions") or {}).keys()),
    }
    out = METADATA_DIR / f"{source_id_value}.info.json"
    out.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    return slim


def confidence(score: float, method: str) -> str:
    if method == "config_spk_n" and score >= 95:
        return "high"
    if score >= 88:
        return "high"
    if score >= 68:
        return "medium"
    if score >= 45:
        return "low"
    return "none"


def build_mapping(refresh_search: bool = False) -> list[dict[str, object]]:
    existing = read_csv(ARG_EXISTING)
    sources = group_sources(existing)
    config_rows = load_config_candidates()
    discovery_rows = load_discovery_candidates()

    rows: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        candidates: list[dict[str, object]] = []
        manifest_url = str(source.get("manifest_url", ""))
        if manifest_url:
            candidates.append(
                {
                    "url": manifest_url,
                    "title": str(source["titulo"]),
                    "channel": "",
                    "duration": "",
                    "method": "existing_manifest_url",
                    "score": 52.0,
                    "raw": {},
                }
            )
        candidates.extend(config_candidates_for(source, config_rows))
        candidates.extend(discovery_candidates_for(source, discovery_rows))

        if not candidates or max(float(c["score"]) for c in candidates) < 68:
            search_cache = METADATA_DIR / f"{source['source_id']}.ytsearch.json"
            if search_cache.exists() and not refresh_search:
                try:
                    candidates.extend(json.loads(search_cache.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    pass
            else:
                search_results = ytsearch_candidates(source)
                serializable = [
                    {key: value for key, value in item.items() if key != "raw"}
                    for item in search_results
                ]
                METADATA_DIR.mkdir(parents=True, exist_ok=True)
                search_cache.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
                candidates.extend(search_results)

        best = max(candidates, key=lambda item: float(item.get("score") or 0.0), default={})
        url = str(best.get("url", ""))
        sid = str(source["source_id"])
        meta = fetch_metadata(url, sid) if url else {}
        candidate_title = str(meta.get("title") or best.get("title", ""))
        candidate_channel = str(meta.get("channel") or best.get("channel", ""))
        candidate_duration = meta.get("duration") or best.get("duration", "")
        method = str(best.get("method", "not_found")) if best else "not_found"
        title_score = similarity(str(source["titulo"]), candidate_title)
        raw_score = float(best.get("score") or 0.0)
        if method.startswith("config_") or method == "existing_manifest_url":
            score = title_score
            if title_score < 45:
                method = f"{method}_metadata_mismatch"
        else:
            score = max(raw_score, title_score)
        conf = confidence(score, method)
        source_status = {
            "high": "mapped_high",
            "medium": "mapped_medium",
            "low": "needs_review",
            "none": "blocked_source_not_found",
        }[conf]
        subtitles = meta.get("subtitles") if isinstance(meta.get("subtitles"), list) else []
        auto_subs = meta.get("automatic_captions") if isinstance(meta.get("automatic_captions"), list) else []
        rows.append(
            {
                **source,
                "candidate_url": meta.get("webpage_url") or url,
                "candidate_video_id": meta.get("id", ""),
                "candidate_title": candidate_title,
                "candidate_channel": candidate_channel,
                "candidate_duration": candidate_duration,
                "match_method": method,
                "match_score": round(score, 2),
                "match_confidence": conf,
                "has_youtube_subs": str(bool(subtitles)).lower(),
                "has_youtube_auto_subs": str(bool(auto_subs)).lower(),
                "source_status": source_status,
                "notes": f"mapping_index={index}; subtitles={','.join(subtitles[:5])}; auto_subs={','.join(auto_subs[:5])}",
            }
        )
        print(f"[{index}/{len(sources)}] {sid}: {conf} {round(score, 1)} {method}")
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row["match_confidence"]) for row in rows)
    status_counts = Counter(str(row["source_status"]) for row in rows)
    clip_counts = Counter()
    for row in rows:
        clip_counts[str(row["match_confidence"])] += int(row.get("clip_count") or 0)
    high_medium = [row for row in rows if row["match_confidence"] in {"high", "medium"}]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Source reconstruction report",
        "",
        f"total_sources: {len(rows)}",
        f"match_confidence_counts: {dict(counts)}",
        f"source_status_counts: {dict(status_counts)}",
        f"clip_counts_by_confidence: {dict(clip_counts)}",
        f"sources_reconstructable_high_medium: {len(high_medium)}",
        f"clips_reconstructable_high_medium: {sum(int(row.get('clip_count') or 0) for row in high_medium)}",
        "",
        "## High/medium mapped sources",
        "",
    ]
    for row in high_medium:
        lines.append(
            "- {source_id}: {match_confidence} score={match_score} clips={clip_count} "
            "method={match_method} url={candidate_url}".format(**row)
        )
    lines.extend(["", "## Low/none sources", ""])
    for row in rows:
        if row["match_confidence"] not in {"high", "medium"}:
            lines.append(
                "- {source_id}: {match_confidence} score={match_score} clips={clip_count} "
                "method={match_method} titulo={titulo}".format(**row)
            )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    refresh_search = "--refresh-search" in argv
    rows = build_mapping(refresh_search=refresh_search)
    write_csv(SOURCE_MAPPING, rows, FIELDS)
    write_report(rows)
    print(f"source_mapping -> {SOURCE_MAPPING}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
