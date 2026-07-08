"""Aplica todos los outputs manuales validados de GPT cleaning."""

from __future__ import annotations

import csv
from pathlib import Path

from apply_gpt_video_patches import apply_patches


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data_cleaning_clean_v1"
HANDOFF = ROOT / "manual_gpt_handoff"
JOB_INDEX = HANDOFF / "job_index.csv"
VALIDATED = ROOT / "validated"


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

    paths = []
    missing = 0
    for row in rows:
        job_id = row.get("job_id", "")
        if not job_id:
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
    print(
        " ".join(
            [
                f"validated_files={len(paths)}",
                f"missing_validated_files={missing}",
                f"applied={result['applied']}",
                f"rejected={result['rejected']}",
                f"train={result['train']}",
                f"eval={result['eval']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
