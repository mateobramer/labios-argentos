"""Genera ROIs 96x96 para clips new_discovery usando MediaPipe existente."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "data_release"
CLIP_MANIFEST = OUT_DIR / "new_discovery_clip_manifest.csv"
ROI_MANIFEST = OUT_DIR / "new_discovery_roi_manifest.csv"
REPORT = OUT_DIR / "reports" / "new_discovery_roi_report.md"
WORK_DIR = OUT_DIR / "work" / "new_discovery_rois"
CLIP_CACHE = WORK_DIR / "clips_cache"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
GCS_ROIS = f"{DEST_BUCKET}/argentina/new_discovery/rois_npz"
GCS_ROI_MP4 = f"{DEST_BUCKET}/argentina/new_discovery/clips_mp4"
GCS_MANIFESTS = f"{DEST_BUCKET}/argentina/new_discovery/manifests"

FIELDS = [
    "dataset_group",
    "video_id",
    "clip_id",
    "clip_name",
    "clip_video_path",
    "clip_video_gcs_path",
    "roi_npz_path",
    "roi_npz_gcs_path",
    "roi_mp4_path",
    "roi_mp4_gcs_path",
    "detect_rate",
    "frames",
    "shape",
    "dtype",
    "status",
    "failure_reason",
    "notes",
]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


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


def write_report(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(r.get("status", "")) for r in rows)
    reasons = Counter(str(r.get("failure_reason", "")) for r in rows if r.get("failure_reason"))
    by_video = Counter(str(r.get("video_id", "")) for r in rows if r.get("status") != "completed_roi")
    lines = [
        "# New discovery ROI report",
        "",
        f"rows: {len(rows)}",
        f"status_counts: {dict(sorted(counts.items()))}",
        f"failure_reason_counts: {dict(sorted(reasons.items()))}",
        "",
        "top_failed_videos:",
        *[f"- {video_id}: {count}" for video_id, count in by_video.most_common(10)],
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def upload_outputs(
    video_id: str,
    roi_dir: Path,
    mp4_dir: Path,
    npz_paths: list[Path] | None = None,
    mp4_paths: list[Path] | None = None,
) -> None:
    gcloud = tool("gcloud")
    subprocess.run([gcloud, "storage", "cp", str(ROI_MANIFEST), f"{GCS_MANIFESTS}/new_discovery_roi_manifest.csv"], check=True)
    subprocess.run([gcloud, "storage", "cp", str(REPORT), f"{DEST_BUCKET}/reports/new_discovery_roi_report.md"], check=True)
    if npz_paths is not None or mp4_paths is not None:
        existing_npz = [str(path) for path in (npz_paths or []) if path.exists()]
        existing_mp4 = [str(path) for path in (mp4_paths or []) if path.exists()]
        if existing_npz:
            subprocess.run([gcloud, "storage", "cp", *existing_npz, f"{GCS_ROIS}/{video_id}/"], check=True)
        if existing_mp4:
            subprocess.run([gcloud, "storage", "cp", *existing_mp4, f"{GCS_ROI_MP4}/{video_id}/"], check=True)
        return
    if roi_dir.exists():
        subprocess.run([gcloud, "storage", "rsync", "--recursive", str(roi_dir), f"{GCS_ROIS}/{video_id}/"], check=True)
    if mp4_dir.exists():
        subprocess.run([gcloud, "storage", "rsync", "--recursive", str(mp4_dir), f"{GCS_ROI_MP4}/{video_id}/"], check=True)


def resolve_clip_path(row: dict[str, str]) -> Path:
    local = Path(row.get("clip_video_path", ""))
    if local.exists():
        return local
    gcs_path = row.get("clip_video_gcs_path", "")
    if not gcs_path:
        return local
    video_id = row.get("video_id", "unknown")
    clip_name = row.get("clip_name", "") or row.get("clip_id", "clip")
    cached = CLIP_CACHE / video_id / f"{clip_name}.mp4"
    if cached.exists():
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("gcloud"), "storage", "cp", gcs_path, str(cached)], check=True)
    return cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--min-detect-rate", type=float, default=0.8)
    parser.add_argument("--review-detect-rate", type=float, default=0.6)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def inspeccionar_video(path: Path) -> dict[str, float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opened": 0.0, "frames": 0.0, "fps": 0.0, "duration": 0.0}
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"opened": 1.0, "frames": frames, "fps": fps, "duration": duration}


def clasificar_bloqueo(
    clip_path: Path,
    ratio: float,
    stats: dict[str, float],
    min_detect_rate: float,
    review_detect_rate: float,
    error: str = "",
) -> tuple[str, str, str]:
    if not clip_path.exists():
        return "blocked_bad_video", "blocked_bad_video", "clip_missing"
    if not stats.get("opened") or stats.get("frames", 0.0) <= 0:
        return "blocked_bad_video", "blocked_bad_video", "video_unreadable_or_empty"
    if stats.get("duration", 0.0) and stats["duration"] < 0.5:
        return "blocked_short_clip", "blocked_short_clip", f"duration={stats['duration']:.3f}"
    if ratio <= 0:
        return "blocked_no_face", "blocked_no_face", "no_face_detected"
    if ratio < review_detect_rate:
        return "blocked_low_face_ratio", "blocked_low_face_ratio", f"detect_rate={ratio:.3f}"
    if ratio < min_detect_rate:
        return "needs_review", "needs_review", f"partial_face_detect_rate={ratio:.3f}"
    if error:
        return "blocked_mouth_not_found", "blocked_mouth_not_found", error[-500:]
    return "blocked_mouth_not_found", "blocked_mouth_not_found", "warp_or_mouth_crop_failed"


def main() -> int:
    args = parse_args()
    import numpy as np

    from visual_preprocessing.src import preprocesar
    from visual_preprocessing.src.preprocesar import crear_landmarker, guardar_npz, guardar_video_gris, procesar_clip
    from visual_preprocessing.src.video_process import VideoProcess

    preprocesar.UMBRAL_DETECCION = args.min_detect_rate
    selected = set(args.video_id)
    clips = [
        row
        for row in read_csv(CLIP_MANIFEST)
        if row.get("status") == "completed_clip_with_audio" and (not selected or row.get("video_id") in selected)
    ]
    if args.limit_clips is not None:
        clips = clips[: args.limit_clips]

    existing = read_csv(ROI_MANIFEST) if args.resume else []
    merged: dict[tuple[str, str], dict[str, object]] = {(r.get("video_id", ""), r.get("clip_id", "")): r for r in existing}
    terminal_statuses = {"completed_roi"} if args.retry_failed else {"completed_roi", "blocked_roi_no_face", "blocked_roi_failed", "blocked_no_face", "blocked_low_face_ratio", "blocked_bad_video", "blocked_short_clip", "blocked_mouth_not_found"}
    done = {key for key, row in merged.items() if row.get("status") in terminal_statuses}

    landmarker = crear_landmarker()
    vproc = VideoProcess(crop_width=96, crop_height=96, convert_gray=True)
    processed = 0
    pending_npz: dict[str, list[Path]] = defaultdict(list)
    pending_mp4: dict[str, list[Path]] = defaultdict(list)
    for row in clips:
        video_id = row.get("video_id", "")
        clip_id = row.get("clip_id", "")
        key = (video_id, clip_id)
        if args.resume and key in done:
            continue
        clip_name = row.get("clip_name", "")
        clip_path = resolve_clip_path(row)
        roi_dir = WORK_DIR / video_id / "rois_npz"
        mp4_dir = WORK_DIR / video_id / "clips_mp4"
        roi_dir.mkdir(parents=True, exist_ok=True)
        mp4_dir.mkdir(parents=True, exist_ok=True)
        roi_path = roi_dir / f"{clip_name}.npz"
        mp4_path = mp4_dir / f"{clip_name}.mp4"
        try:
            frames, ratio = procesar_clip(str(clip_path), landmarker, vproc)
            if frames:
                guardar_npz(frames, str(roi_path))
                guardar_video_gris(frames, str(mp4_path))
                pending_npz[video_id].append(roi_path)
                pending_mp4[video_id].append(mp4_path)
                arr = np.asarray(frames, dtype=np.uint8)
                status = "completed_roi"
                failure_reason = ""
                notes = ""
                shape = "x".join(str(x) for x in arr.shape)
                dtype = str(arr.dtype)
            else:
                stats = inspeccionar_video(clip_path)
                status, failure_reason, notes = clasificar_bloqueo(
                    clip_path,
                    ratio,
                    stats,
                    args.min_detect_rate,
                    args.review_detect_rate,
                )
                shape = ""
                dtype = ""
        except Exception as exc:  # noqa: BLE001
            ratio = 0.0
            frames = []
            stats = inspeccionar_video(clip_path)
            status, failure_reason, notes = clasificar_bloqueo(
                clip_path,
                ratio,
                stats,
                args.min_detect_rate,
                args.review_detect_rate,
                str(exc),
            )
            shape = ""
            dtype = ""
        merged[key] = {
            "dataset_group": "argentina/new_discovery",
            "video_id": video_id,
            "clip_id": clip_id,
            "clip_name": clip_name,
            "clip_video_path": str(clip_path),
            "clip_video_gcs_path": row.get("clip_video_gcs_path", ""),
            "roi_npz_path": str(roi_path) if roi_path.exists() else "",
            "roi_npz_gcs_path": f"{GCS_ROIS}/{video_id}/{clip_name}.npz" if roi_path.exists() else "",
            "roi_mp4_path": str(mp4_path) if mp4_path.exists() else "",
            "roi_mp4_gcs_path": f"{GCS_ROI_MP4}/{video_id}/{clip_name}.mp4" if mp4_path.exists() else "",
            "detect_rate": f"{ratio:.3f}",
            "frames": len(frames),
            "shape": shape,
            "dtype": dtype,
            "status": status,
            "failure_reason": failure_reason,
            "notes": notes,
        }
        processed += 1
        if processed == 1 or processed % max(1, args.checkpoint_every) == 0:
            rows = list(merged.values())
            write_csv(ROI_MANIFEST, rows, FIELDS)
            write_report(rows)
            print(f"checkpoint processed={processed} rows_total={len(rows)}", flush=True)
            if args.upload:
                upload_outputs(video_id, roi_dir, mp4_dir, pending_npz[video_id], pending_mp4[video_id])
                pending_npz[video_id] = []
                pending_mp4[video_id] = []
    rows = list(merged.values())
    write_csv(ROI_MANIFEST, rows, FIELDS)
    write_report(rows)
    if args.upload:
        for video_id in sorted({row.get("video_id", "") for row in clips if row.get("video_id")}):
            upload_outputs(video_id, WORK_DIR / video_id / "rois_npz", WORK_DIR / video_id / "clips_mp4")
    print(f"manifest -> {ROI_MANIFEST}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
