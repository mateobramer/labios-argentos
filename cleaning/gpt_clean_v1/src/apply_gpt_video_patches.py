"""Aplica patches GPT validados a los manifests finales."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "cleaning/gpt_clean_v1"
RELEASE = REPO / "data_release"
VALIDATED = ROOT / "validated"
TRANSCRIPTS = ROOT / "clean_gpt_v1"
JOBS = ROOT / "video_jobs"
HANDOFF = ROOT / "manual_gpt_handoff"
MANUAL_INDEX = HANDOFF / "job_index.csv"
PATCH_LOG = ROOT / "patch_log.csv"
REJECTED = ROOT / "rejected_patches.jsonl"

FINAL_RELEASE = RELEASE / "final_release_manifest.csv"
FINAL_TRAIN = RELEASE / "final_train_manifest_clean_gpt_v1.csv"
FINAL_EVAL = RELEASE / "final_eval_manifest_clean_gpt_v1.csv"
CLEAN_GPT = RELEASE / "clean_gpt_manifest.csv"
REPORT = RELEASE / "reports" / "gpt_cleaning_report.md"

FINAL_FIELDS = [
    "dataset_group",
    "source_id",
    "clip_id",
    "split",
    "spk",
    "titulo",
    "source_url",
    "source_video_id",
    "start_time",
    "end_time",
    "mp4_visual_roi_path",
    "npz_path",
    "clip_with_audio_path",
    "existing_text",
    "large_text",
    "turbo_text",
    "clean_gpt_text",
    "selected_training_text",
    "text_source",
    "clean_status",
    "clean_confidence",
    "patch_count",
    "alignment_confidence",
    "asr_status",
    "gpt_status",
    "usable_for_training",
    "usable_for_eval",
    "needs_review",
    "failure_reason",
    "notes",
]

CLEAN_FIELDS = [
    "dataset_group",
    "source_id",
    "clip_id",
    "existing_text",
    "large_text",
    "turbo_text",
    "clean_text",
    "status",
    "confidence",
    "patch_count",
    "gpt_status",
    "notes",
]

PATCH_FIELDS = [
    "applied_at",
    "video_id",
    "clip_id",
    "action",
    "confidence",
    "patch_count",
    "text_before",
    "text_after",
    "reason",
    "notes",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "unknown_video"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main_clip_ids_for_job(job_id: str) -> set[str]:
    input_path = JOBS / f"{job_id}_input.jsonl"
    if not input_path.exists():
        return set()
    return {
        row.get("clip_id", "")
        for row in read_jsonl(input_path)
        if row.get("clip_id") and not truthy(row.get("context_only"))
    }


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_fallback(row: dict[str, str]) -> tuple[str, str]:
    if row.get("large_text", "").strip():
        return row["large_text"], "large"
    if row.get("turbo_text", "").strip():
        return row["turbo_text"], "turbo"
    return row.get("existing_text", ""), "existing_text"


def transcript_path(video_id: str, clip_id: str) -> Path:
    clip_name = safe_name(clip_id.split("::")[-1] if "::" in clip_id else clip_id)
    return TRANSCRIPTS / safe_name(video_id) / f"{clip_name}.txt"


def load_validated(paths: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    patches = {}
    rejected = []
    for path in paths:
        job_id = path.stem.removesuffix("_validated")
        video_id = job_id.removeprefix("video_")
        main_clip_ids = main_clip_ids_for_job(job_id)
        for row in read_jsonl(path):
            action = row.get("action")
            clip_id = row.get("clip_id", "")
            if main_clip_ids and clip_id not in main_clip_ids:
                continue
            if action == "reject":
                rejected.append({"video_id": video_id, "clip_id": clip_id, "record": row, "reason": "action_reject"})
                continue
            patches[clip_id] = {**row, "_video_id": video_id}
    return patches, rejected


def apply_patches(paths: list[Path]) -> dict[str, int]:
    final_rows = read_csv(FINAL_RELEASE)
    clean_rows = {row.get("clip_id", ""): row for row in read_csv(CLEAN_GPT)}
    patches, rejected = load_validated(paths)
    rejected_by_clip = {row.get("clip_id", ""): row for row in rejected}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    patch_log_rows = []
    applied = 0

    for row in final_rows:
        clip_id = row.get("clip_id", "")
        patch = patches.get(clip_id)
        fallback_text, fallback_source = selected_fallback(row)
        if patch:
            clean_text = str(patch.get("clean_text", "")).strip()
            if clean_text:
                action = patch.get("action", "")
                patch_count = 1 if action == "patch" and clean_text != fallback_text else 0
                video_id = row.get("source_video_id", "") or patch.get("_video_id", "")
                out_path = transcript_path(video_id, clip_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(clean_text + "\n", encoding="utf-8", newline="\n")
                row["clean_gpt_text"] = clean_text
                row["selected_training_text"] = clean_text
                row["text_source"] = "clean_gpt_v1"
                row["clean_status"] = "completed_clean_gpt"
                row["clean_confidence"] = patch.get("confidence", "medium")
                row["patch_count"] = patch_count
                row["gpt_status"] = "completed_clean_gpt"
                row["needs_review"] = "false"
                if row.get("usable_for_training") != "true" and not row.get("npz_path"):
                    row["failure_reason"] = "pending_new_discovery_roi_npz"
                elif row.get("usable_for_training") == "true":
                    row["failure_reason"] = ""
                row["notes"] = "gpt_cleaning_applied"
                clean_rows[clip_id] = {
                    "dataset_group": row.get("dataset_group", ""),
                    "source_id": row.get("source_id", ""),
                    "clip_id": clip_id,
                    "existing_text": row.get("existing_text", ""),
                    "large_text": row.get("large_text", ""),
                    "turbo_text": row.get("turbo_text", ""),
                    "clean_text": clean_text,
                    "status": "completed_clean_gpt",
                    "confidence": patch.get("confidence", "medium"),
                    "patch_count": patch_count,
                    "gpt_status": "completed_clean_gpt",
                    "notes": patch.get("reason", ""),
                }
                patch_log_rows.append(
                    {
                        "applied_at": now,
                        "video_id": video_id,
                        "clip_id": clip_id,
                        "action": action,
                        "confidence": patch.get("confidence", ""),
                        "patch_count": patch_count,
                        "text_before": fallback_text,
                        "text_after": clean_text,
                        "reason": patch.get("reason", ""),
                        "notes": patch.get("notes", ""),
                    }
                )
                applied += 1
        else:
            row["selected_training_text"] = row.get("selected_training_text") or fallback_text
            row["text_source"] = row.get("text_source") or fallback_source
            rejected_patch = rejected_by_clip.get(clip_id)
            if rejected_patch:
                clean_rows[clip_id] = {
                    "dataset_group": row.get("dataset_group", ""),
                    "source_id": row.get("source_id", ""),
                    "clip_id": clip_id,
                    "existing_text": row.get("existing_text", ""),
                    "large_text": row.get("large_text", ""),
                    "turbo_text": row.get("turbo_text", ""),
                    "clean_text": "",
                    "status": "rejected_clean_gpt",
                    "confidence": rejected_patch.get("record", {}).get("confidence", "low"),
                    "patch_count": 0,
                    "gpt_status": "rejected_clean_gpt",
                    "notes": rejected_patch.get("record", {}).get("reason", "action_reject"),
                }
                patch_log_rows.append(
                    {
                        "applied_at": now,
                        "video_id": row.get("source_video_id", ""),
                        "clip_id": clip_id,
                        "action": "reject",
                        "confidence": rejected_patch.get("record", {}).get("confidence", ""),
                        "patch_count": 0,
                        "text_before": fallback_text,
                        "text_after": "",
                        "reason": rejected_patch.get("record", {}).get("reason", "action_reject"),
                        "notes": rejected_patch.get("record", {}).get("notes", ""),
                    }
                )

    train_rows = [r for r in final_rows if r.get("split") == "train" and r.get("usable_for_training") == "true"]
    eval_rows = [r for r in final_rows if r.get("split") in {"val", "test"} and r.get("usable_for_eval") == "true"]
    write_csv(FINAL_RELEASE, final_rows, FINAL_FIELDS)
    write_csv(FINAL_TRAIN, train_rows, FINAL_FIELDS)
    write_csv(FINAL_EVAL, eval_rows, FINAL_FIELDS)
    write_csv(CLEAN_GPT, clean_rows.values(), CLEAN_FIELDS)
    previous_patch_rows = read_csv(PATCH_LOG)
    write_csv(PATCH_LOG, [*previous_patch_rows, *patch_log_rows], PATCH_FIELDS)
    append_jsonl(REJECTED, rejected)
    counts = Counter(row.get("clean_status", "") for row in final_rows)
    no_roi_cleaned = sum(
        1
        for row in final_rows
        if row.get("clean_status") == "completed_clean_gpt" and row.get("usable_for_training") != "true"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# GPT cleaning report",
                "",
                f"applied_at: {now}",
                f"validated_files: {len(paths)}",
                f"completed_clean_gpt: {counts.get('completed_clean_gpt', 0)}",
                f"completed_large_turbo_no_gpt: {counts.get('completed_large_turbo_no_gpt', 0)}",
                f"patches_rejected: {len(rejected)}",
                f"no_roi_but_text_cleaned: {no_roi_cleaned}",
                f"train_rows: {len(train_rows)}",
                f"eval_rows: {len(eval_rows)}",
                "",
                "Regla visual: `usable_for_training` no se cambia por GPT; sigue dependiendo de ROI valido.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"applied": applied, "rejected": len(rejected), "train": len(train_rows), "eval": len(eval_rows)}


def paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.path:
        return [Path(p) for p in args.path]
    if args.all:
        return [
            VALIDATED / f"{row.get('job_id', '')}_validated.jsonl"
            for row in read_csv(MANUAL_INDEX)
            if row.get("job_id")
        ]
    if args.job_id:
        return [VALIDATED / f"{job_id}_validated.jsonl" for job_id in args.job_id]
    if args.video_id:
        return [VALIDATED / f"video_{safe_name(video_id)}_validated.jsonl" for video_id in args.video_id]
    return sorted(VALIDATED.glob("video_*_validated.jsonl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [path for path in paths_from_args(args) if path.exists()]
    if not paths:
        print("No hay JSONL validado para aplicar.")
        return 1
    result = apply_patches(paths)
    print(f"applied={result['applied']} rejected={result['rejected']} train={result['train']} eval={result['eval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
