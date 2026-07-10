"""Alinea clips existing contra subtitulos de YouTube cuando existen.

Esta etapa no corre Whisper ni descarga videos completos. Produce timestamps
candidatos para fuentes high/medium con subtitulos automaticos disponibles.
"""

from __future__ import annotations

import csv
import html
import re
import shutil
import subprocess
import sys
import unicodedata
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_pipeline/release"
METADATA_DIR = OUT_DIR / "source_metadata"
SUBTITLE_DIR = METADATA_DIR / "subtitles"
REPORTS_DIR = OUT_DIR / "reports"

SOURCE_MAPPING = OUT_DIR / "source_mapping.csv"
ARG_EXISTING = OUT_DIR / "argentina_existing_manifest.csv"
ALIGNMENT = OUT_DIR / "alignment_manifest.csv"
REPORT = REPORTS_DIR / "alignment_report.md"
PROGRESS_LOG = REPORTS_DIR / "alignment_progress.log"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"

FIELDS = [
    "source_id",
    "clip_id",
    "split",
    "titulo",
    "spk",
    "existing_text",
    "source_url",
    "source_video_id",
    "start_time",
    "end_time",
    "expected_duration",
    "extracted_duration",
    "alignment_method",
    "alignment_score",
    "alignment_confidence",
    "large_full_segment_text",
    "turbo_full_segment_text",
    "clip_audio_path",
    "clip_video_path",
    "status",
    "notes",
]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


YTDLP = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")


def ytdlp_cmd() -> list[str]:
    if YTDLP:
        return [YTDLP]
    return [sys.executable, "-m", "yt_dlp"]


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def norm(text: str) -> str:
    value = html.unescape(text or "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9ñ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_time(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def strip_vtt_text(text: str) -> str:
    text = re.sub(r"<\d\d:\d\d:\d\d\.\d+>", " ", text)
    text = re.sub(r"</?c[^>]*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def new_suffix(previous: list[str], current: list[str]) -> list[str]:
    if not previous:
        return current
    limit = min(len(previous), len(current), 30)
    for overlap in range(limit, 0, -1):
        if previous[-overlap:] == current[:overlap]:
            return current[overlap:]
    # YouTube VTT a veces repite la linea previa completa y agrega palabras al final.
    for start in range(min(len(current), 30)):
        candidate = current[start:]
        if candidate and previous[-min(len(previous), len(candidate), 10) :] != candidate[: min(len(previous), len(candidate), 10)]:
            return candidate
    return current


def parse_vtt(path: Path) -> list[dict[str, object]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", raw)
    segments: list[dict[str, object]] = []
    previous_words: list[str] = []
    timing_re = re.compile(r"(\d\d:\d\d:\d\d\.\d+|\d\d:\d\d\.\d+)\s+-->\s+(\d\d:\d\d:\d\d\.\d+|\d\d:\d\d\.\d+)")
    for block in blocks:
        match = timing_re.search(block)
        if not match:
            continue
        start = parse_time(match.group(1))
        end = parse_time(match.group(2))
        lines = block.splitlines()
        text_lines = []
        after_timing = False
        for line in lines:
            if timing_re.search(line):
                after_timing = True
                continue
            if not after_timing:
                continue
            if line.strip() and not line.strip().startswith(("NOTE", "WEBVTT", "Kind:", "Language:")):
                text_lines.append(line)
        text = strip_vtt_text(" ".join(text_lines))
        words = norm(text).split()
        suffix = new_suffix(previous_words, words)
        previous_words.extend(suffix)
        if suffix:
            text_value = " ".join(suffix)
            segments.append({"start": start, "end": end, "text": text_value, "norm_text": norm(text_value)})
    return segments


def score_norm(na: str, nb: str) -> float:
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    aset = set(na.split())
    bset = set(nb.split())
    overlap = len(aset & bset) / max(1, len(aset | bset))
    return round((0.7 * ratio + 0.3 * overlap) * 100, 2)


def score_text(a: str, b: str) -> float:
    return score_norm(norm(a), norm(b))


def confidence(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 65:
        return "medium"
    if score >= 50:
        return "low"
    return "none"


def clip_number(clip: str) -> int:
    match = re.search(r"(\d+)", clip or "")
    return int(match.group(1)) if match else 0


def log_progress(message: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


def download_subtitles(source: dict[str, str], timeout: int) -> list[Path]:
    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    sid = source["source_id"]
    existing = sorted(SUBTITLE_DIR.glob(f"{sid}*.vtt"))
    if existing:
        return existing
    url = source.get("candidate_url", "")
    if not url:
        return []
    output = str(SUBTITLE_DIR / f"{sid}.%(ext)s")
    args = ytdlp_cmd() + [
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs",
        "es.*,es-419,es",
        "--sub-format",
        "vtt",
        "--socket-timeout",
        "20",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        "--output",
        output,
        url,
    ]
    result = run(args, timeout=timeout)
    if result.returncode != 0 and not list(SUBTITLE_DIR.glob(f"{sid}*.vtt")):
        return []
    return sorted(SUBTITLE_DIR.glob(f"{sid}*.vtt"))


def best_subtitle(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    for suffix in [".es-orig.vtt", ".es.vtt", ".es-419.vtt"]:
        for path in paths:
            if path.name.endswith(suffix):
                return path
    return paths[0]


def align_clip(clip: dict[str, str], segments: list[dict[str, object]], start_index: int) -> tuple[dict[str, object], int]:
    expected = max(0.1, float(clip.get("n_frames") or 0) / 25.0)
    target = clip.get("text_large_existing", "")
    target_norm = norm(target)
    target_words = set(target_norm.split())
    best: dict[str, object] = {"score": 0.0, "i": start_index, "j": start_index, "text": ""}
    max_window = min(40.0, max(8.0, expected * 3.0 + 4.0))
    scan_span = 350 if start_index == 0 else 120
    max_start = min(len(segments), start_index + scan_span)
    for i in range(start_index, max_start):
        start = float(segments[i]["start"])
        texts: list[str] = []
        norm_texts: list[str] = []
        for j in range(i, min(len(segments), i + 40)):
            end = float(segments[j]["end"])
            if end - start > max_window:
                break
            texts.append(str(segments[j]["text"]))
            norm_texts.append(str(segments[j].get("norm_text", norm(str(segments[j]["text"])))))
            candidate = " ".join(texts)
            candidate_norm = " ".join(norm_texts)
            candidate_words = set(candidate_norm.split())
            token_recall = len(target_words & candidate_words) / max(1, len(target_words))
            score = token_recall * 45.0 if token_recall < 0.2 else score_norm(target_norm, candidate_norm)
            duration = end - start
            duration_penalty = min(20.0, abs(duration - expected) / max(expected, 0.5) * 10.0)
            adjusted = score - duration_penalty
            if adjusted > float(best["score"]):
                best = {
                    "score": round(adjusted, 2),
                    "raw_score": score,
                    "i": i,
                    "j": j,
                    "start": start,
                    "end": end,
                    "text": candidate,
                    "duration": duration,
                }
        if i > start_index + 20 and float(best["score"]) >= 75:
            break
        if i > start_index + 80 and float(best["score"]) >= 65:
            break
    conf = confidence(float(best.get("score") or 0.0))
    next_index = min(len(segments) - 1, int(best.get("j", start_index)) + 1) if segments else start_index
    if conf == "none":
        next_index = start_index
    return {
        "start_time": round(float(best.get("start", 0.0)), 3) if conf != "none" else "",
        "end_time": round(float(best.get("end", 0.0)), 3) if conf != "none" else "",
        "extracted_duration": round(float(best.get("duration", 0.0)), 3) if conf != "none" else "",
        "alignment_score": best.get("score", 0.0),
        "alignment_confidence": conf,
        "large_full_segment_text": best.get("text", "") if conf != "none" else "",
        "status": "needs_review" if conf in {"high", "medium", "low"} else "blocked_alignment_failed",
        "notes": f"subtitle_raw_score={best.get('raw_score','')}; pending_clip_extraction_asr_gpt",
    }, next_index


def build_alignment(args: argparse.Namespace) -> list[dict[str, object]]:
    mapping_rows = {row["source_id"]: row for row in read_csv(SOURCE_MAPPING)}
    clips_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(ARG_EXISTING):
        sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row.get('spk')}__{row.get('titulo')}").strip("_")[:120]
        clips_by_source[sid].append(row)

    rows: list[dict[str, object]] = []
    processed_sources: set[str] = set()
    if args.resume and ALIGNMENT.exists():
        rows = list(read_csv(ALIGNMENT))
        processed_sources = {str(row["source_id"]) for row in rows}
        log_progress(f"resume: loaded_rows={len(rows)} processed_sources={len(processed_sources)}")

    selected_sources = set(args.source or [])
    processed_now = 0
    for sid, clips in sorted(clips_by_source.items()):
        if selected_sources and sid not in selected_sources:
            continue
        if args.resume and sid in processed_sources:
            continue
        if args.limit is not None and processed_now >= args.limit:
            break
        source = mapping_rows.get(sid, {})
        clips_sorted = sorted(clips, key=lambda item: clip_number(item.get("clip", "")))
        should_try_subs = source.get("match_confidence") in {"high", "medium"} and source.get("has_youtube_auto_subs") == "true"
        segments: list[dict[str, object]] = []
        subtitle_path = None
        log_progress(f"source_start source_id={sid} clips={len(clips_sorted)} try_subs={should_try_subs}")
        if should_try_subs:
            subtitle_path = best_subtitle(download_subtitles(source, args.download_timeout))
            if subtitle_path:
                segments = parse_vtt(subtitle_path)
        cursor = 0
        for clip in clips_sorted:
            expected_duration = round(float(clip.get("n_frames") or 0) / 25.0, 3)
            base = {
                "source_id": sid,
                "clip_id": clip.get("clip_id", ""),
                "split": clip.get("split", ""),
                "titulo": clip.get("titulo", ""),
                "spk": clip.get("spk", ""),
                "existing_text": clip.get("text_large_existing", ""),
                "source_url": source.get("candidate_url", ""),
                "source_video_id": source.get("candidate_video_id", ""),
                "expected_duration": expected_duration,
                "alignment_method": "youtube_auto_subs_fuzzy_ordered" if segments else "not_attempted_no_subtitles",
                "turbo_full_segment_text": "",
                "clip_audio_path": "",
                "clip_video_path": "",
            }
            if segments:
                aligned, cursor = align_clip(clip, segments, cursor)
                rows.append({**base, **aligned})
            else:
                reason = "no_youtube_auto_subs" if source.get("match_confidence") in {"high", "medium"} else "source_not_high_medium"
                rows.append(
                    {
                        **base,
                        "start_time": "",
                        "end_time": "",
                        "extracted_duration": "",
                        "alignment_score": "",
                        "alignment_confidence": "none",
                        "large_full_segment_text": "",
                        "status": "needs_review" if source.get("match_confidence") in {"high", "medium"} else "blocked_source_not_found",
                        "notes": f"{reason}; requires_full_audio_asr_alignment",
                    }
                )
        processed_now += 1
        write_csv(ALIGNMENT, rows, FIELDS)
        write_report(rows)
        log_progress(f"source_done source_id={sid} clips={len(clips_sorted)} subtitles={bool(subtitle_path)} segments={len(segments)} rows_total={len(rows)}")
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    confidence_counts = Counter(str(row["alignment_confidence"]) for row in rows)
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    source_summary = []
    for sid, items in by_source.items():
        scores = [float(row["alignment_score"]) for row in items if str(row.get("alignment_score", "")).strip()]
        source_summary.append(
            {
                "source_id": sid,
                "clips": len(items),
                "aligned": sum(1 for row in items if row["alignment_confidence"] in {"high", "medium", "low"}),
                "avg_score": round(mean(scores), 2) if scores else "",
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alignment report",
        "",
        f"rows: {len(rows)}",
        f"status_counts: {dict(status_counts)}",
        f"alignment_confidence_counts: {dict(confidence_counts)}",
        "",
        "## Sources with aligned clips",
        "",
    ]
    for item in sorted(source_summary, key=lambda row: int(row["aligned"]), reverse=True):
        if item["aligned"]:
            lines.append(f"- {item['source_id']}: aligned={item['aligned']}/{item['clips']} avg_score={item['avg_score']}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reusar alignment_manifest.csv y omitir fuentes ya procesadas.")
    parser.add_argument("--limit", type=int, default=None, help="Procesar como maximo N fuentes en esta corrida.")
    parser.add_argument("--source", action="append", help="Procesar solo un source_id; se puede repetir.")
    parser.add_argument("--download-timeout", type=int, default=90, help="Timeout por descarga de subtitulos en segundos.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_alignment(args)
    write_csv(ALIGNMENT, rows, FIELDS)
    write_report(rows)
    print(f"alignment_manifest -> {ALIGNMENT}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
