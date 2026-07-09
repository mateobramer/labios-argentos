"""Aplica todos los outputs manuales validados de GPT cleaning."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from apply_gpt_video_patches import apply_patches


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data_cleaning_clean_v1"
HANDOFF = ROOT / "manual_gpt_handoff"
JOB_INDEX = HANDOFF / "job_index.csv"
VALIDATED = ROOT / "validated"
REPORTS = ROOT / "reports"
VALIDATION_REPORT = REPORTS / "manual_gpt_validation_report.csv"
APPLY_REPORT = REPORTS / "manual_gpt_apply_report.md"


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

    validation_rows = {row.get("job_id", ""): row for row in read_csv(VALIDATION_REPORT)}
    paths = []
    missing = 0
    skipped_failed = 0
    skipped_missing_raw = 0
    for row in rows:
        job_id = row.get("job_id", "")
        if not job_id:
            continue
        validation = validation_rows.get(job_id, {})
        status = validation.get("status", "")
        if status == "missing_raw_output":
            skipped_missing_raw += 1
            continue
        if validation_rows and status != "validated":
            skipped_failed += 1
            continue
        path = VALIDATED / f"{job_id}_validated.jsonl"
        if path.exists():
            paths.append(path)
        else:
            missing += 1

    if not paths:
        print("No hay JSONL validado para aplicar.")
        return 1

    result = apply_patches(paths)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    REPORTS.mkdir(parents=True, exist_ok=True)
    APPLY_REPORT.write_text(
        "\n".join(
            [
                "# Manual GPT apply report",
                "",
                f"applied_at: {now}",
                f"validated_files_applied: {len(paths)}",
                f"missing_validated_files: {missing}",
                f"skipped_failed_jobs: {skipped_failed}",
                f"skipped_missing_raw_outputs: {skipped_missing_raw}",
                f"applied_rows: {result['applied']}",
                f"rejected_rows: {result['rejected']}",
                f"train_rows: {result['train']}",
                f"eval_rows: {result['eval']}",
                "",
                "Solo se aplicaron jobs con status `validated` en `data_cleaning_clean_v1/reports/manual_gpt_validation_report.csv`.",
                "Los jobs incompletos, JSONL invalidos o con outputs rechazados quedan documentados en el reporte de validacion.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        " ".join(
            [
                f"validated_files={len(paths)}",
                f"missing_validated_files={missing}",
                f"skipped_failed_jobs={skipped_failed}",
                f"skipped_missing_raw_outputs={skipped_missing_raw}",
                f"applied={result['applied']}",
                f"rejected={result['rejected']}",
                f"train={result['train']}",
                f"eval={result['eval']}",
                f"report={APPLY_REPORT}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
