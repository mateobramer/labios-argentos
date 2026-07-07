"""Transcribe clips new_discovery con faster-whisper."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_release"
CLIP_MANIFEST = OUT_DIR / "new_discovery_clip_manifest.csv"
ASR_MANIFEST = OUT_DIR / "new_discovery_asr_manifest.csv"
REPORT = OUT_DIR / "reports" / "new_discovery_asr_report.md"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
GCS_TRANSCRIPTS = f"{DEST_BUCKET}/argentina/new_discovery/transcripts"
GCS_MANIFESTS = f"{DEST_BUCKET}/argentina/new_discovery/manifests"
WORK_DIR = OUT_DIR / "work" / "new_discovery_asr"
CLIP_CACHE = WORK_DIR / "clips_cache"

FIELDS = [
    "dataset_group",
    "video_id",
    "clip_id",
    "clip_name",
    "model_role",
    "model_name",
    "device",
    "compute_type",
    "clip_video_path",
    "clip_video_gcs_path",
    "transcript_path",
    "transcript_gcs_path",
    "text",
    "language",
    "duration",
    "status",
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


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--model debe ser rol=modelo")
    role, model = value.split("=", 1)
    return role.strip(), model.strip()


def write_report(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(r.get("status", "")) for r in rows)
    by_role = Counter(str(r.get("model_role", "")) for r in rows)
    lines = [
        "# New discovery ASR report",
        "",
        f"rows: {len(rows)}",
        f"status_counts: {dict(sorted(counts.items()))}",
        f"rows_by_role: {dict(sorted(by_role.items()))}",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def upload_outputs(role: str, video_id: str, transcript_dir: Path) -> None:
    gcloud = tool("gcloud")
    subprocess.run([gcloud, "storage", "cp", str(ASR_MANIFEST), f"{GCS_MANIFESTS}/new_discovery_asr_manifest.csv"], check=True)
    subprocess.run([gcloud, "storage", "cp", str(REPORT), f"{DEST_BUCKET}/reports/new_discovery_asr_report.md"], check=True)
    if transcript_dir.exists():
        subprocess.run(
            [gcloud, "storage", "rsync", "--recursive", str(transcript_dir), f"{GCS_TRANSCRIPTS}/{role}/{video_id}/"],
            check=True,
        )


def transcribe(model, clip: Path, beam_size: int) -> tuple[str, str, float]:
    segments, info = model.transcribe(str(clip), language="es", beam_size=beam_size, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    return text, getattr(info, "language", "") or "", float(getattr(info, "duration", 0.0) or 0.0)


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
    parser.add_argument("--model", action="append", required=True, help="Formato rol=modelo; repetible.")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        write_report([{"status": "blocked_missing_asr_dependency", "notes": str(exc), "model_role": "", "model_name": ""}])
        raise

    selected = set(args.video_id)
    clips = [
        row
        for row in read_csv(CLIP_MANIFEST)
        if row.get("status") == "completed_clip_with_audio" and (not selected or row.get("video_id") in selected)
    ]
    if args.limit_clips is not None:
        clips = clips[: args.limit_clips]

    existing = read_csv(ASR_MANIFEST) if args.resume else []
    merged: dict[tuple[str, str, str], dict[str, object]] = {
        (row.get("video_id", ""), row.get("clip_id", ""), row.get("model_role", "")): row for row in existing
    }

    for role, model_name in [parse_model(v) for v in args.model]:
        print(f"model_start role={role} model={model_name}", flush=True)
        model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)
        processed = 0
        for row in clips:
            key = (row.get("video_id", ""), row.get("clip_id", ""), role)
            if args.resume and merged.get(key, {}).get("status") == "completed_asr":
                continue
            video_id = row.get("video_id", "")
            clip_name = row.get("clip_name", "")
            clip_path = resolve_clip_path(row)
            transcript_dir = WORK_DIR / "transcripts" / role / video_id
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{clip_name}.txt"
            try:
                text, language, duration = transcribe(model, clip_path, args.beam_size)
                transcript_path.write_text(text + "\n", encoding="utf-8")
                status = "completed_asr" if text else "needs_review_empty_asr"
                notes = ""
            except Exception as exc:  # noqa: BLE001
                text, language, duration = "", "", 0.0
                status = "blocked_asr_failed"
                notes = str(exc)[-500:]
            merged[key] = {
                "dataset_group": "argentina/new_discovery",
                "video_id": video_id,
                "clip_id": row.get("clip_id", ""),
                "clip_name": clip_name,
                "model_role": role,
                "model_name": model_name,
                "device": args.device,
                "compute_type": args.compute_type,
                "clip_video_path": str(clip_path),
                "clip_video_gcs_path": row.get("clip_video_gcs_path", ""),
                "transcript_path": str(transcript_path),
                "transcript_gcs_path": f"{GCS_TRANSCRIPTS}/{role}/{video_id}/{clip_name}.txt",
                "text": text,
                "language": language,
                "duration": round(duration, 3),
                "status": status,
                "notes": notes,
            }
            processed += 1
            if processed == 1 or processed % max(1, args.checkpoint_every) == 0:
                all_rows = list(merged.values())
                write_csv(ASR_MANIFEST, all_rows, FIELDS)
                write_report(all_rows)
                print(f"checkpoint role={role} processed={processed} rows_total={len(all_rows)}", flush=True)
                if args.upload:
                    upload_outputs(role, video_id, transcript_dir)
        all_rows = list(merged.values())
        write_csv(ASR_MANIFEST, all_rows, FIELDS)
        write_report(all_rows)
        if args.upload:
            for video_id in sorted({r.get("video_id", "") for r in clips if r.get("video_id")}):
                upload_outputs(role, video_id, WORK_DIR / "transcripts" / role / video_id)
        print(f"model_done role={role} rows_total={len(all_rows)}", flush=True)

    print(f"manifest -> {ASR_MANIFEST}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
