"""Valida todos los outputs manuales de GPT cleaning."""

from __future__ import annotations

import csv
from pathlib import Path

from validate_gpt_video_jsonl import REPORT_FIELDS, validate_video, write_csv


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "cleaning/gpt_clean_v1"
HANDOFF = ROOT / "manual_gpt_handoff"
JOB_INDEX = HANDOFF / "job_index.csv"
RAW = ROOT / "raw_outputs"
REPORTS = HANDOFF / "reports"
REPORT = REPORTS / "manual_gpt_validation_report.csv"
ROOT_REPORTS = ROOT / "reports"
ROOT_REPORT = ROOT_REPORTS / "manual_gpt_validation_report.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = read_csv(JOB_INDEX)
    if not rows:
        print(f"No existe job_index manual: {JOB_INDEX}")
        return 1

    REPORTS.mkdir(parents=True, exist_ok=True)
    report_rows = []
    missing = 0
    failed = 0
    validated = 0

    for row in rows:
        job_id = row.get("job_id", "")
        if not job_id:
            continue
        raw_path = RAW / f"{job_id}_raw.jsonl"
        if not raw_path.exists():
            missing += 1
            report_rows.append(
                {
                    "job_id": job_id,
                    "video_id": row.get("video_id", ""),
                    "input_path": row.get("prompt_md_path", ""),
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
            validated += 1
        else:
            failed += 1

    write_csv(REPORT, report_rows, REPORT_FIELDS)
    write_csv(ROOT_REPORT, report_rows, REPORT_FIELDS)
    print(f"validated_jobs={validated} failed_jobs={failed} missing_raw_outputs={missing}")
    print(f"report -> {REPORT}")
    print(f"report -> {ROOT_REPORT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
