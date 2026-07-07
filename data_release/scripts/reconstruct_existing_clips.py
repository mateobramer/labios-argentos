"""Reconstruye clips argentinos existing con audio desde videos fuente.

Lee `alignment_manifest.csv` y extrae intervalos high/medium desde la fuente
original. El trabajo es reanudable por clip y puede subir cada source a GCS.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_release"
WORK_DIR = OUT_DIR / "work" / "existing_reconstruction"

ALIGNMENT = OUT_DIR / "alignment_manifest.csv"
SOURCE_MAPPING = OUT_DIR / "source_mapping.csv"
MANIFEST = OUT_DIR / "existing_reconstruction_manifest.csv"
REPORT = OUT_DIR / "reports" / "existing_reconstruction_report.md"
FAILURES = OUT_DIR / "reports" / "failures.csv"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
GCS_CLIPS = f"{DEST_BUCKET}/argentina/existing/clips_with_audio"
GCS_AUDIO = f"{DEST_BUCKET}/argentina/existing/reconstructed_audio"
GCS_MANIFESTS = f"{DEST_BUCKET}/argentina/existing/manifests"

FIELDS = [
    "source_id",
    "clip_id",
    "clip_name",
    "source_url",
    "source_video_id",
    "start_time",
    "end_time",
    "expected_duration",
    "extracted_duration",
    "alignment_confidence",
    "clip_video_path",
    "clip_audio_path",
    "clip_video_gcs_path",
    "clip_audio_gcs_path",
    "has_audio",
    "status",
    "notes",
]

FAILURE_FIELDS = ["stage", "dataset_group", "source_id", "clip_id", "path", "error_type", "error_message", "notes"]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


YTDLP = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
GCLOUD = shutil.which("gcloud.cmd") or shutil.which("gcloud")


def ytdlp_cmd() -> list[str]:
    if YTDLP:
        return [YTDLP]
    return [sys.executable, "-m", "yt_dlp"]


def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def run_ok(args: list[str], timeout: int = 900) -> str:
    result = run(args, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Fallo comando: {' '.join(args)}\n{result.stderr[-3000:]}")
    return result.stdout + result.stderr


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_failure(row: dict[str, object]) -> None:
    existing = read_csv(FAILURES)
    existing.append(row)
    write_csv(FAILURES, existing, FAILURE_FIELDS)


def clip_name(clip_id: str) -> str:
    match = re.search(r"(clip_\d+)", clip_id or "")
    if match:
        return match.group(1)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip_id or "clip")
    return safe.strip("_")[:80]


def load_rows(confidences: set[str], sources: set[str]) -> dict[str, list[dict[str, str]]]:
    rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(ALIGNMENT):
        if sources and row.get("source_id") not in sources:
            continue
        if row.get("alignment_confidence") not in confidences:
            continue
        if not row.get("start_time") or not row.get("end_time"):
            continue
        rows_by_source[row["source_id"]].append(row)
    return rows_by_source


def download_source(source_id: str, url: str, work: Path, fmt: str, timeout: int) -> Path:
    existing = sorted([p for p in work.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm"}])
    if existing:
        return existing[0]
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path()
    output = str(work / "source.%(ext)s")
    args = ytdlp_cmd() + [
        "--no-playlist",
        "--no-part",
        "--windows-filenames",
        "--merge-output-format",
        "mp4",
        "--ffmpeg-location",
        ffmpeg,
        "-f",
        fmt,
        "-o",
        output,
        url,
    ]
    run_ok(args, timeout=timeout)
    candidates = sorted([p for p in work.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm"}])
    if not candidates:
        raise RuntimeError("yt-dlp no genero video fuente")
    return candidates[0]


def extract_clip(source_video: Path, row: dict[str, str], source_work: Path) -> dict[str, object]:
    ffmpeg = ffmpeg_path()
    name = clip_name(row.get("clip_id", ""))
    clips_dir = source_work / "clips_with_audio"
    audio_dir = source_work / "reconstructed_audio"
    clips_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clips_dir / f"{name}.mp4"
    audio_path = audio_dir / f"{name}.wav"
    start = float(row["start_time"])
    end = float(row["end_time"])
    duration = max(0.05, end - start)

    if not clip_path.exists():
        run_ok(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                str(clip_path),
            ],
            timeout=180,
        )
    if not audio_path.exists():
        run_ok(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            timeout=180,
        )

    probe = run([ffmpeg, "-hide_banner", "-i", str(clip_path), "-f", "null", "-"], timeout=120)
    combined = probe.stdout + probe.stderr
    has_audio = "Audio:" in combined and "audio:0KiB" not in combined
    status = "completed_reconstructed_audio" if has_audio else "blocked_no_audio"
    return {
        "source_id": row.get("source_id", ""),
        "clip_id": row.get("clip_id", ""),
        "clip_name": name,
        "source_url": row.get("source_url", ""),
        "source_video_id": row.get("source_video_id", ""),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "expected_duration": row.get("expected_duration", ""),
        "extracted_duration": round(duration, 3),
        "alignment_confidence": row.get("alignment_confidence", ""),
        "clip_video_path": str(clip_path),
        "clip_audio_path": str(audio_path),
        "clip_video_gcs_path": f"{GCS_CLIPS}/{row.get('source_id','')}/{name}.mp4",
        "clip_audio_gcs_path": f"{GCS_AUDIO}/{row.get('source_id','')}/{name}.wav",
        "has_audio": str(has_audio).lower(),
        "status": status,
        "notes": "extracted_from_mapped_youtube_source",
    }


def upload_source(source_id: str, source_work: Path) -> None:
    if not GCLOUD:
        raise RuntimeError("No se encontro gcloud")
    clips_dir = source_work / "clips_with_audio"
    audio_dir = source_work / "reconstructed_audio"
    if clips_dir.exists():
        run_ok([GCLOUD, "storage", "cp", "--recursive", str(clips_dir / "*"), f"{GCS_CLIPS}/{source_id}/"], timeout=1800)
    if audio_dir.exists():
        run_ok([GCLOUD, "storage", "cp", "--recursive", str(audio_dir / "*"), f"{GCS_AUDIO}/{source_id}/"], timeout=1800)
    run_ok([GCLOUD, "storage", "cp", str(MANIFEST), f"{GCS_MANIFESTS}/existing_reconstruction_manifest.csv"], timeout=300)


def cleanup_source_work(path: Path) -> None:
    resolved = path.resolve()
    root = WORK_DIR.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Ruta de cleanup fuera de work dir: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def write_report(rows: list[dict[str, object]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status", ""))] += 1
    sources = {str(row.get("source_id", "")) for row in rows if row.get("source_id")}
    lines = [
        "# Existing reconstruction report",
        "",
        f"rows: {len(rows)}",
        f"sources: {len(sources)}",
        f"status_counts: {dict(sorted(counts.items()))}",
        "",
        "Los clips reconstruidos vienen de timestamps high/medium en alignment_manifest.csv.",
        "Los outputs pesados se guardan en GCS y no se versionan en git.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", default=[], help="Procesar solo este source_id; repetible.")
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--max-clips-per-source", type=int, default=None)
    parser.add_argument("--confidence", action="append", default=["high", "medium"])
    parser.add_argument("--format", default="bv*[height<=720]+ba/b[height<=720]/b")
    parser.add_argument("--download-timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--cleanup-source-work", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    confidences = set(args.confidence)
    selected_sources = set(args.source)
    rows_by_source = load_rows(confidences, selected_sources)
    existing_rows = read_csv(MANIFEST) if args.resume else []
    done = {(row.get("source_id", ""), row.get("clip_id", "")) for row in existing_rows if row.get("status") == "completed_reconstructed_audio"}
    merged: dict[tuple[str, str], dict[str, object]] = {(row.get("source_id", ""), row.get("clip_id", "")): row for row in existing_rows}
    processed_sources = 0

    for source_id, rows in sorted(rows_by_source.items()):
        if args.limit_sources is not None and processed_sources >= args.limit_sources:
            break
        source_rows = rows[: args.max_clips_per_source] if args.max_clips_per_source else rows
        if args.resume and source_rows and all((source_id, row.get("clip_id", "")) in done for row in source_rows):
            continue
        source_url = source_rows[0].get("source_url", "")
        if not source_url:
            continue
        source_work = WORK_DIR / source_id
        print(f"source_start source_id={source_id} clips={len(source_rows)}", flush=True)
        try:
            source_video = download_source(source_id, source_url, source_work, args.format, args.download_timeout)
            for row in source_rows:
                key = (source_id, row.get("clip_id", ""))
                if args.resume and key in done:
                    continue
                try:
                    merged[key] = extract_clip(source_video, row, source_work)
                except Exception as exc:  # noqa: BLE001 - registro por clip y sigo.
                    merged[key] = {
                        "source_id": source_id,
                        "clip_id": row.get("clip_id", ""),
                        "clip_name": clip_name(row.get("clip_id", "")),
                        "source_url": source_url,
                        "source_video_id": row.get("source_video_id", ""),
                        "start_time": row.get("start_time", ""),
                        "end_time": row.get("end_time", ""),
                        "expected_duration": row.get("expected_duration", ""),
                        "extracted_duration": "",
                        "alignment_confidence": row.get("alignment_confidence", ""),
                        "status": "blocked_reconstruction_failed",
                        "has_audio": "false",
                        "notes": str(exc)[-500:],
                    }
                    append_failure(
                        {
                            "stage": "existing_reconstruction",
                            "dataset_group": "argentina/existing",
                            "source_id": source_id,
                            "clip_id": row.get("clip_id", ""),
                            "path": source_url,
                            "error_type": "clip_extract_failed",
                            "error_message": str(exc)[-1000:],
                            "notes": "continuing_with_next_clip",
                        }
                    )
            all_rows = list(merged.values())
            write_csv(MANIFEST, all_rows, FIELDS)
            write_report(all_rows)
            if args.upload:
                upload_source(source_id, source_work)
            if args.cleanup_source_work:
                cleanup_source_work(source_work)
            processed_sources += 1
            print(f"source_done source_id={source_id} rows_total={len(all_rows)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - registro por source y sigo.
            append_failure(
                {
                    "stage": "existing_reconstruction",
                    "dataset_group": "argentina/existing",
                    "source_id": source_id,
                    "clip_id": "",
                    "path": source_url,
                    "error_type": "source_reconstruction_failed",
                    "error_message": str(exc)[-1000:],
                    "notes": "continuing_with_next_source",
                }
            )
            print(f"source_failed source_id={source_id} error={str(exc)[-300:]}", flush=True)

    all_rows = list(merged.values())
    write_csv(MANIFEST, all_rows, FIELDS)
    write_report(all_rows)
    print(f"manifest -> {MANIFEST}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
