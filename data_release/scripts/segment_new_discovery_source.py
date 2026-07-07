"""Segmenta fuentes new_discovery descargadas en clips con audio.

Usa video/audio ya descargados localmente por `download_sources_local.py`.
No descarga de YouTube ni usa cookies. El resultado es reanudable por clip y
se sube incrementalmente a GCS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_release"
WORK_DIR = OUT_DIR / "work" / "new_discovery_ingest"
DOWNLOAD_MANIFEST = OUT_DIR / "local_source_download_manifest.csv"
SEGMENT_MANIFEST = OUT_DIR / "new_discovery_clip_manifest.csv"
REPORT = OUT_DIR / "reports" / "new_discovery_ingest_report.md"
FAILURES = OUT_DIR / "reports" / "failures.csv"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
GCS_CLIPS = f"{DEST_BUCKET}/argentina/new_discovery/clips_with_audio"
GCS_MANIFESTS = f"{DEST_BUCKET}/argentina/new_discovery/manifests"
GCS_REPORTS = f"{DEST_BUCKET}/reports"

FIELDS = [
    "dataset_group",
    "video_id",
    "source_id",
    "clip_id",
    "clip_name",
    "source_video_gcs_path",
    "source_audio_gcs_path",
    "start_time",
    "end_time",
    "duration",
    "clip_video_path",
    "clip_video_gcs_path",
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


def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def run_ok(args: list[str], timeout: int = 900) -> str:
    result = run(args, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Fallo comando: {' '.join(args)}\n{result.stderr[-2000:]}")
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
    rows = read_csv(FAILURES)
    rows.append(row)
    write_csv(FAILURES, rows, FAILURE_FIELDS)


def media_duration(path: Path) -> float:
    ffmpeg = ffmpeg_path()
    result = run([ffmpeg, "-hide_banner", "-i", str(path)], timeout=60)
    text = result.stdout + result.stderr
    import re

    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        raise RuntimeError(f"No se pudo leer duracion de {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def has_audio(path: Path) -> bool:
    ffmpeg = ffmpeg_path()
    result = run([ffmpeg, "-hide_banner", "-i", str(path)], timeout=60)
    text = result.stdout + result.stderr
    return "Audio:" in text


def make_windows(duration: float, clip_seconds: float, start_offset: float, end_margin: float) -> list[tuple[float, float]]:
    end_limit = max(start_offset, duration - end_margin)
    windows = []
    n = int(math.floor((end_limit - start_offset) / clip_seconds))
    for idx in range(n):
        start = start_offset + idx * clip_seconds
        end = min(start + clip_seconds, end_limit)
        if end - start >= 2.0:
            windows.append((start, end))
    return windows


def extract_clip(video: Path, audio: Path, out: Path, start: float, duration: float) -> None:
    ffmpeg = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ok(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-shortest",
            str(out),
        ],
        timeout=180,
    )


def upload_source_outputs(video_id: str, source_work: Path) -> None:
    gcloud = tool("gcloud")
    clips_dir = source_work / "clips_with_audio"
    if clips_dir.exists():
        run_ok([gcloud, "storage", "cp", "--recursive", str(clips_dir / "*.mp4"), f"{GCS_CLIPS}/{video_id}/"], timeout=1800)
    run_ok([gcloud, "storage", "cp", str(SEGMENT_MANIFEST), f"{GCS_MANIFESTS}/new_discovery_clip_manifest.csv"], timeout=300)
    run_ok([gcloud, "storage", "cp", str(REPORT), f"{GCS_REPORTS}/new_discovery_ingest_report.md"], timeout=300)


def write_report(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(r.get("status", "")) for r in rows)
    videos = sorted({str(r.get("video_id", "")) for r in rows if r.get("video_id")})
    lines = [
        "# New discovery ingest report",
        "",
        f"clip_rows: {len(rows)}",
        f"videos: {len(videos)}",
        f"status_counts: {dict(sorted(counts.items()))}",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--clip-seconds", type=float, default=6.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--end-margin", type=float, default=0.0)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-each-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.video_id)
    downloads = [
        row
        for row in read_csv(DOWNLOAD_MANIFEST)
        if row.get("status", "").startswith("downloaded_uploaded")
        and row.get("local_video_path")
        and row.get("local_audio_path")
        and (not selected or row.get("source_id") in selected)
    ]
    existing = read_csv(SEGMENT_MANIFEST) if args.resume else []
    merged: dict[tuple[str, str], dict[str, object]] = {(r.get("video_id", ""), r.get("clip_id", "")): r for r in existing}
    done = {key for key, row in merged.items() if row.get("status") == "completed_clip_with_audio"}

    for source in downloads:
        video_id = source.get("source_id", "")
        video = Path(source.get("local_video_path", ""))
        audio = Path(source.get("local_audio_path", ""))
        source_work = WORK_DIR / video_id
        duration = min(media_duration(video), media_duration(audio))
        windows = make_windows(duration, args.clip_seconds, args.start_offset, args.end_margin)
        if args.limit_clips is not None:
            windows = windows[: args.limit_clips]
        processed = 0
        print(f"source_start video_id={video_id} windows={len(windows)} duration={duration:.1f}", flush=True)
        for idx, (start, end) in enumerate(windows):
            clip_name = f"clip_{idx:04d}"
            clip_id = f"new_discovery::{video_id}::{clip_name}"
            key = (video_id, clip_id)
            if args.resume and key in done:
                continue
            out = source_work / "clips_with_audio" / f"{clip_name}.mp4"
            try:
                if not out.exists():
                    extract_clip(video, audio, out, start, end - start)
                ok_audio = has_audio(out)
                status = "completed_clip_with_audio" if ok_audio else "blocked_no_audio"
                merged[key] = {
                    "dataset_group": "argentina/new_discovery",
                    "video_id": video_id,
                    "source_id": video_id,
                    "clip_id": clip_id,
                    "clip_name": clip_name,
                    "source_video_gcs_path": source.get("gcs_video_path", ""),
                    "source_audio_gcs_path": source.get("gcs_audio_path", ""),
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "duration": round(end - start, 3),
                    "clip_video_path": str(out),
                    "clip_video_gcs_path": f"{GCS_CLIPS}/{video_id}/{clip_name}.mp4",
                    "has_audio": str(ok_audio).lower(),
                    "status": status,
                    "notes": "fixed_window_from_downloaded_source",
                }
            except Exception as exc:  # noqa: BLE001
                merged[key] = {
                    "dataset_group": "argentina/new_discovery",
                    "video_id": video_id,
                    "source_id": video_id,
                    "clip_id": clip_id,
                    "clip_name": clip_name,
                    "source_video_gcs_path": source.get("gcs_video_path", ""),
                    "source_audio_gcs_path": source.get("gcs_audio_path", ""),
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "duration": round(end - start, 3),
                    "clip_video_path": str(out),
                    "clip_video_gcs_path": "",
                    "has_audio": "false",
                    "status": "blocked_clip_extract_failed",
                    "notes": str(exc)[-500:],
                }
                append_failure(
                    {
                        "stage": "new_discovery_clip_extract",
                        "dataset_group": "argentina/new_discovery",
                        "source_id": video_id,
                        "clip_id": clip_id,
                        "path": str(out),
                        "error_type": "clip_extract_failed",
                        "error_message": str(exc)[-1000:],
                        "notes": "continuing_with_next_clip",
                    }
                )
            processed += 1
            if processed == 1 or processed % max(1, args.checkpoint_every) == 0:
                rows = list(merged.values())
                write_csv(SEGMENT_MANIFEST, rows, FIELDS)
                write_report(rows)
                print(f"checkpoint video_id={video_id} processed={processed} rows_total={len(rows)}", flush=True)
                if args.upload and args.upload_each_checkpoint:
                    upload_source_outputs(video_id, source_work)
        rows = list(merged.values())
        write_csv(SEGMENT_MANIFEST, rows, FIELDS)
        write_report(rows)
        if args.upload:
            upload_source_outputs(video_id, source_work)
        print(f"source_done video_id={video_id} rows_total={len(rows)}", flush=True)

    rows = list(merged.values())
    write_csv(SEGMENT_MANIFEST, rows, FIELDS)
    write_report(rows)
    print(f"manifest -> {SEGMENT_MANIFEST}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
