"""Transcribe clips reconstruidos con faster-whisper.

Acepta pares rol=modelo, por ejemplo:
  --model large=large-v3 --model turbo=large-v3-turbo
o para smoke liviano:
  --model sample_small=small
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_pipeline/release"
WORK_DIR = OUT_DIR / "work" / "asr_existing"
RECON_MANIFEST = OUT_DIR / "existing_reconstruction_manifest.csv"
ASR_MANIFEST = OUT_DIR / "asr_large_turbo_manifest.csv"
DISAGREEMENT = OUT_DIR / "asr_disagreement_v2.csv"
REPORT = OUT_DIR / "reports" / "asr_large_turbo_report.md"
FAILURES = OUT_DIR / "reports" / "failures.csv"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
GCS_TRANSCRIPTS = f"{DEST_BUCKET}/argentina/existing/transcripts"
GCS_MANIFESTS = f"{DEST_BUCKET}/argentina/existing/manifests"

FIELDS = [
    "dataset_group",
    "source_id",
    "clip_id",
    "clip_name",
    "model_role",
    "model_name",
    "device",
    "compute_type",
    "audio_path",
    "audio_gcs_path",
    "transcript_path",
    "transcript_gcs_path",
    "text",
    "language",
    "duration",
    "status",
    "notes",
]

DISAGREEMENT_FIELDS = [
    "source_id",
    "clip_id",
    "clip_name",
    "large_text",
    "turbo_text",
    "cer",
    "wer",
    "length_ratio",
    "status",
    "notes",
]

FAILURE_FIELDS = ["stage", "dataset_group", "source_id", "clip_id", "path", "error_type", "error_message", "notes"]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


GCLOUD = shutil.which("gcloud.cmd") or shutil.which("gcloud")


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
    existing = read_csv(FAILURES)
    existing.append(row)
    write_csv(FAILURES, existing, FAILURE_FIELDS)


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--model debe tener formato rol=modelo")
    role, model = value.split("=", 1)
    return role.strip(), model.strip()


def norm(text: str) -> str:
    text = (text or "").casefold()
    text = re.sub(r"[^a-z0-9áéíóúüñ\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    return round(edit_distance(list(na), list(nb)) / max(1, len(na)), 4)


def wer(a: str, b: str) -> float:
    wa, wb = norm(a).split(), norm(b).split()
    return round(edit_distance(wa, wb) / max(1, len(wa)), 4)


def local_audio_path(row: dict[str, str]) -> Path:
    path = Path(row.get("clip_audio_path", ""))
    if path.exists():
        return path
    source_id = row.get("source_id", "")
    clip_name = row.get("clip_name", "")
    local = WORK_DIR / "audio_cache" / source_id / f"{clip_name}.wav"
    if local.exists():
        return local
    gcs_path = row.get("clip_audio_gcs_path", "")
    if not gcs_path or not GCLOUD:
        raise RuntimeError("audio local no existe y no hay GCS path/gcloud para descargar")
    local.parent.mkdir(parents=True, exist_ok=True)
    run_ok([GCLOUD, "storage", "cp", gcs_path, str(local)], timeout=300)
    return local


def transcribe_audio(model, audio: Path, beam_size: int) -> tuple[str, str, float]:
    segments, info = model.transcribe(str(audio), language="es", beam_size=beam_size, vad_filter=True)
    parts = []
    for segment in segments:
        parts.append(segment.text.strip())
    text = " ".join(part for part in parts if part).strip()
    language = getattr(info, "language", "") or ""
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return text, language, duration


def upload_outputs(role: str, source_id: str, transcript_dir: Path) -> None:
    if not GCLOUD:
        raise RuntimeError("No se encontro gcloud")
    run_ok([GCLOUD, "storage", "cp", str(ASR_MANIFEST), f"{GCS_MANIFESTS}/asr_large_turbo_manifest.csv"], timeout=300)
    if DISAGREEMENT.exists():
        run_ok([GCLOUD, "storage", "cp", str(DISAGREEMENT), f"{GCS_MANIFESTS}/asr_disagreement_v2.csv"], timeout=300)
    if transcript_dir.exists():
        run_ok(
            [GCLOUD, "storage", "cp", "--recursive", str(transcript_dir / "*.txt"), f"{GCS_TRANSCRIPTS}/{role}/{source_id}/"],
            timeout=600,
        )
    run_ok([GCLOUD, "storage", "cp", str(REPORT), f"{DEST_BUCKET}/reports/asr_large_turbo_report.md"], timeout=300)


def write_disagreement(asr_rows: list[dict[str, object]]) -> None:
    by_clip: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in asr_rows:
        if row.get("status") == "completed_asr":
            by_clip[(str(row.get("source_id", "")), str(row.get("clip_id", "")))][str(row.get("model_role", ""))] = row
    rows = []
    for (source_id, clip_id), items in sorted(by_clip.items()):
        if "large" not in items or "turbo" not in items:
            continue
        large = str(items["large"].get("text", ""))
        turbo = str(items["turbo"].get("text", ""))
        length_ratio = round(len(norm(turbo)) / max(1, len(norm(large))), 4)
        rows.append(
            {
                "source_id": source_id,
                "clip_id": clip_id,
                "clip_name": items["large"].get("clip_name", ""),
                "large_text": large,
                "turbo_text": turbo,
                "cer": cer(large, turbo),
                "wer": wer(large, turbo),
                "length_ratio": length_ratio,
                "status": "computed",
                "notes": "",
            }
        )
    write_csv(DISAGREEMENT, rows, DISAGREEMENT_FIELDS)


def write_report(rows: list[dict[str, object]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    models: dict[str, str] = {}
    for row in rows:
        counts[str(row.get("status", ""))] += 1
        by_role[str(row.get("model_role", ""))] += 1
        models[str(row.get("model_role", ""))] = str(row.get("model_name", ""))
    lines = [
        "# ASR large/turbo report",
        "",
        f"rows: {len(rows)}",
        f"status_counts: {dict(sorted(counts.items()))}",
        f"rows_by_role: {dict(sorted(by_role.items()))}",
        f"models_by_role: {dict(sorted(models.items()))}",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="Formato rol=modelo; repetible.")
    parser.add_argument("--source", action="append", default=[])
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
        write_report(
            [
                {
                    "status": "blocked_missing_asr_dependency",
                    "notes": str(exc),
                    "model_role": "",
                    "model_name": "",
                }
            ]
        )
        raise

    selected_sources = set(args.source)
    recon_rows = [
        row
        for row in read_csv(RECON_MANIFEST)
        if row.get("status") == "completed_reconstructed_audio" and (not selected_sources or row.get("source_id") in selected_sources)
    ]
    if args.limit_clips is not None:
        recon_rows = recon_rows[: args.limit_clips]

    existing = read_csv(ASR_MANIFEST) if args.resume else []
    merged: dict[tuple[str, str, str], dict[str, object]] = {
        (row.get("source_id", ""), row.get("clip_id", ""), row.get("model_role", "")): row for row in existing
    }

    models = [parse_model(value) for value in args.model]
    for role, model_name in models:
        print(f"model_start role={role} model={model_name}", flush=True)
        try:
            model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)
        except Exception as exc:  # noqa: BLE001
            for row in recon_rows:
                key = (row.get("source_id", ""), row.get("clip_id", ""), role)
                merged[key] = {
                    "dataset_group": "argentina/existing",
                    "source_id": row.get("source_id", ""),
                    "clip_id": row.get("clip_id", ""),
                    "clip_name": row.get("clip_name", ""),
                    "model_role": role,
                    "model_name": model_name,
                    "device": args.device,
                    "compute_type": args.compute_type,
                    "audio_path": row.get("clip_audio_path", ""),
                    "audio_gcs_path": row.get("clip_audio_gcs_path", ""),
                    "status": "blocked_asr_model_load_failed",
                    "notes": str(exc)[-500:],
                }
            continue
        processed_for_role = 0
        for row in recon_rows:
            key = (row.get("source_id", ""), row.get("clip_id", ""), role)
            if args.resume and merged.get(key, {}).get("status") == "completed_asr":
                continue
            source_id = row.get("source_id", "")
            clip_name = row.get("clip_name", "")
            transcript_dir = WORK_DIR / "transcripts" / role / source_id
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{clip_name}.txt"
            try:
                audio = local_audio_path(row)
                text, language, duration = transcribe_audio(model, audio, args.beam_size)
                transcript_path.write_text(text + "\n", encoding="utf-8")
                merged[key] = {
                    "dataset_group": "argentina/existing",
                    "source_id": source_id,
                    "clip_id": row.get("clip_id", ""),
                    "clip_name": clip_name,
                    "model_role": role,
                    "model_name": model_name,
                    "device": args.device,
                    "compute_type": args.compute_type,
                    "audio_path": str(audio),
                    "audio_gcs_path": row.get("clip_audio_gcs_path", ""),
                    "transcript_path": str(transcript_path),
                    "transcript_gcs_path": f"{GCS_TRANSCRIPTS}/{role}/{source_id}/{clip_name}.txt",
                    "text": text,
                    "language": language,
                    "duration": round(duration, 3),
                    "status": "completed_asr" if text else "needs_review_empty_asr",
                    "notes": "",
                }
            except Exception as exc:  # noqa: BLE001
                merged[key] = {
                    "dataset_group": "argentina/existing",
                    "source_id": source_id,
                    "clip_id": row.get("clip_id", ""),
                    "clip_name": clip_name,
                    "model_role": role,
                    "model_name": model_name,
                    "device": args.device,
                    "compute_type": args.compute_type,
                    "audio_path": row.get("clip_audio_path", ""),
                    "audio_gcs_path": row.get("clip_audio_gcs_path", ""),
                    "status": "blocked_asr_failed",
                    "notes": str(exc)[-500:],
                }
                append_failure(
                    {
                        "stage": "asr_existing",
                        "dataset_group": "argentina/existing",
                        "source_id": source_id,
                        "clip_id": row.get("clip_id", ""),
                        "path": row.get("clip_audio_gcs_path", row.get("clip_audio_path", "")),
                        "error_type": "asr_failed",
                        "error_message": str(exc)[-1000:],
                        "notes": f"role={role}; model={model_name}",
                    }
                )
            processed_for_role += 1
            if processed_for_role == 1 or processed_for_role % max(1, args.checkpoint_every) == 0:
                all_rows = list(merged.values())
                write_csv(ASR_MANIFEST, all_rows, FIELDS)
                write_disagreement(all_rows)
                write_report(all_rows)
                print(
                    f"checkpoint role={role} processed_for_role={processed_for_role} rows_total={len(all_rows)}",
                    flush=True,
                )
                if args.upload:
                    upload_outputs(role, source_id, transcript_dir)
        all_rows = list(merged.values())
        write_csv(ASR_MANIFEST, all_rows, FIELDS)
        write_disagreement(all_rows)
        write_report(all_rows)
        if args.upload:
            for source_id in sorted({row.get("source_id", "") for row in recon_rows if row.get("source_id")}):
                upload_outputs(role, source_id, WORK_DIR / "transcripts" / role / source_id)
        print(f"model_done role={role} rows_total={len(all_rows)}", flush=True)

    all_rows = list(merged.values())
    write_csv(ASR_MANIFEST, all_rows, FIELDS)
    write_disagreement(all_rows)
    write_report(all_rows)
    print(f"manifest -> {ASR_MANIFEST}")
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
