"""Valida salidas JSONL de GPT cleaning por video."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data_cleaning_clean_v1"
JOBS = ROOT / "video_jobs"
RAW = ROOT / "raw_outputs"
VALIDATED = ROOT / "validated"
REPORT = VALIDATED / "validation_report.csv"
HANDOFF = ROOT / "manual_gpt_handoff"
MANUAL_INDEX = HANDOFF / "job_index.csv"
MANUAL_REPORT = HANDOFF / "reports" / "manual_gpt_validation_report.csv"

VALID_ACTIONS = {"keep", "patch", "reject"}
VALID_CONFIDENCE = {"high", "medium", "low"}
GENERIC_TEXTS = {
    "no se entiende",
    "no entiendo",
    "inaudible",
    "transcripcion no disponible",
    "texto no disponible",
    "no aplica",
}

REPORT_FIELDS = [
    "job_id",
    "video_id",
    "input_path",
    "raw_path",
    "validated_path",
    "rejected_path",
    "input_count",
    "raw_count",
    "validated_count",
    "rejected_count",
    "missing_count",
    "invented_count",
    "status",
    "notes",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "unknown_video"


def read_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    records = []
    rejected = []
    if not path.exists():
        return records, [{"line": 0, "error": "raw_output_missing", "raw": ""}]
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                rejected.append({"line": line_no, "error": f"invalid_json:{exc}", "raw": raw})
                continue
            if not isinstance(item, dict):
                rejected.append({"line": line_no, "error": "json_item_not_object", "raw": raw})
                continue
            item["_line"] = line_no
            records.append(item)
    return records, rejected


def read_input(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                out[item["clip_id"]] = item
    return out


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "_line"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def word_count(text: str) -> int:
    return len(str(text or "").split())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_clip_id(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def validate_record(record: dict, input_by_clip: dict[str, dict], clip_aliases: dict[str, str]) -> tuple[bool, list[str]]:
    errors = []
    clip_id = str(record.get("clip_id", "")).strip()
    if not clip_id:
        errors.append("missing_clip_id")
        return False, errors
    if clip_id not in input_by_clip:
        canonical = clip_aliases.get(normalize_clip_id(clip_id))
        if canonical:
            record["clip_id"] = canonical
            clip_id = canonical
        else:
            errors.append("invented_clip_id")
            return False, errors

    for field in ["action", "clean_text", "reason", "confidence", "notes"]:
        if field not in record:
            errors.append(f"missing_{field}")
    action = record.get("action")
    if action not in VALID_ACTIONS:
        errors.append("invalid_action")
    if record.get("confidence") not in VALID_CONFIDENCE:
        errors.append("invalid_confidence")

    source = input_by_clip[clip_id]
    if truthy(source.get("context_only")):
        errors.append("output_on_context_only")
        return False, errors
    selected = str(source.get("selected_asr_text") or source.get("large_text") or source.get("turbo_text") or "")
    clean_text = str(record.get("clean_text", "")).strip()
    if action in {"keep", "patch"} and not clean_text:
        errors.append("empty_clean_text")
    if action == "reject":
        return not errors, errors
    if normalize(clean_text) in GENERIC_TEXTS:
        errors.append("generic_clean_text")

    selected_words = word_count(selected)
    clean_words = word_count(clean_text)
    if selected_words >= 6:
        if clean_words < max(1, int(selected_words * 0.55)):
            errors.append("clean_text_deletes_too_much")
        if clean_words > max(60, int(selected_words * 1.8) + 10):
            errors.append("clean_text_too_long")
    if action == "patch" and selected and clean_text:
        ratio = SequenceMatcher(None, normalize(selected), normalize(clean_text)).ratio()
        if ratio < 0.35:
            errors.append("patch_too_different")

    for field in ["start_time", "end_time"]:
        if field in record and str(record.get(field, "")) != str(source.get(field, "")):
            errors.append(f"changed_{field}")
    return not errors, errors


def resolve_paths(job_id: str, raw_path: str | None) -> tuple[Path, Path, Path, Path]:
    input_path = JOBS / f"{job_id}_input.jsonl"
    raw = Path(raw_path) if raw_path else RAW / f"{job_id}_raw.jsonl"
    validated = VALIDATED / f"{job_id}_validated.jsonl"
    rejected = VALIDATED / f"{job_id}_rejected.jsonl"
    return input_path, raw, validated, rejected


def validate_video(job_id: str, raw_path: str | None = None, video_id: str = "") -> dict[str, object]:
    input_path, raw, validated_path, rejected_path = resolve_paths(job_id, raw_path)
    input_by_clip = read_input(input_path)
    clip_aliases = {normalize_clip_id(clip_id): clip_id for clip_id in input_by_clip}
    main_clip_ids = {clip_id for clip_id, row in input_by_clip.items() if not truthy(row.get("context_only"))}
    raw_records, parse_rejections = read_jsonl(raw)
    valid_rows = []
    rejected_rows = list(parse_rejections)
    seen = set()
    invented = 0
    for record in raw_records:
        ok, errors = validate_record(record, input_by_clip, clip_aliases)
        clip_id = str(record.get("clip_id", "")).strip()
        if clip_id in seen:
            ok = False
            errors.append("duplicate_clip_id")
        if clip_id:
            seen.add(clip_id)
        if "invented_clip_id" in errors:
            invented += 1
        if ok:
            valid_rows.append(record)
        else:
            rejected_rows.append({"clip_id": clip_id, "errors": errors, "record": record})

    missing = sorted(main_clip_ids - {str(row.get("clip_id", "")).strip() for row in raw_records})
    for clip_id in missing:
        rejected_rows.append({"clip_id": clip_id, "errors": ["missing_output"], "record": input_by_clip[clip_id]})

    write_jsonl(validated_path, valid_rows)
    write_jsonl(rejected_path, rejected_rows)
    status = "validated" if valid_rows and not parse_rejections else "failed_jsonl"
    if missing:
        status = "failed_jsonl"
    return {
        "job_id": job_id,
        "video_id": video_id,
        "input_path": str(input_path),
        "raw_path": str(raw),
        "validated_path": str(validated_path),
        "rejected_path": str(rejected_path),
        "input_count": len(input_by_clip),
        "raw_count": len(raw_records),
        "validated_count": len(valid_rows),
        "rejected_count": len(rejected_rows),
        "missing_count": len(missing),
        "invented_count": invented,
        "status": status,
        "notes": "",
    }


def update_report(row: dict[str, object]) -> None:
    rows = [r for r in read_csv(REPORT) if r.get("job_id") != row.get("job_id")]
    rows.append(row)
    write_csv(REPORT, rows, REPORT_FIELDS)


def validate_all_manual() -> dict[str, int]:
    rows = read_csv(MANUAL_INDEX)
    report_rows = []
    counts = {"validated": 0, "failed": 0, "missing": 0}
    for row in rows:
        job_id = row.get("job_id", "")
        if not job_id:
            continue
        raw_path = RAW / f"{job_id}_raw.jsonl"
        if not raw_path.exists():
            counts["missing"] += 1
            report_rows.append(
                {
                    "job_id": job_id,
                    "video_id": row.get("video_id", ""),
                    "input_path": str(JOBS / f"{job_id}_input.jsonl"),
                    "raw_path": str(raw_path),
                    "validated_path": "",
                    "rejected_path": "",
                    "input_count": row.get("main_clips", ""),
                    "raw_count": 0,
                    "validated_count": 0,
                    "rejected_count": 0,
                    "missing_count": row.get("main_clips", ""),
                    "invented_count": 0,
                    "status": "missing_raw_output",
                    "notes": "raw_outputs/<job_id>_raw.jsonl no existe",
                }
            )
            continue
        result = validate_video(job_id, str(raw_path), row.get("video_id", ""))
        report_rows.append(result)
        if result["status"] == "validated":
            counts["validated"] += 1
        else:
            counts["failed"] += 1
    write_csv(MANUAL_REPORT, report_rows, REPORT_FIELDS)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        counts = validate_all_manual()
        print(
            " ".join(
                [
                    f"validated_jobs={counts['validated']}",
                    f"failed_jobs={counts['failed']}",
                    f"missing_raw_outputs={counts['missing']}",
                    f"report={MANUAL_REPORT}",
                ]
            )
        )
        return 1 if counts["failed"] else 0
    job_id = args.job_id or f"video_{safe_name(args.video_id or '')}"
    row = validate_video(job_id, args.raw_path, args.video_id or "")
    update_report(row)
    print(f"{row['status']}: validated={row['validated_count']} rejected={row['rejected_count']} -> {row['validated_path']}")
    return 0 if row["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
