"""Parsea salidas VSR batch a CSV estandarizado."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from evaluation.src.experiment_metrics import cer, wer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
RESULT_COLUMNS = [
    "experiment",
    "source_id",
    "clip",
    "split",
    "training_usability",
    "policy_moderate",
    "preprocessing_variant",
    "transcript_variant",
    "reference",
    "hypothesis",
    "wer",
    "cer",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def leer_config(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def leer_inf(path: Path) -> list[tuple[str, str]]:
    pares = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "#" not in line:
            raise ValueError(f"Linea sin separador '#': {line[:80]}")
        ref, hyp = line.split("#", 1)
        pares.append((ref.strip(), hyp.strip()))
    return pares


def _metadata_rows(config: dict[str, str], raw_experiment_dir: Path) -> list[dict[str, str]]:
    mapeo = raw_experiment_dir / "test_mapeo.csv"
    if mapeo.exists():
        rows = []
        for row in leer_csv(mapeo):
            rows.append(
                {
                    "split": "test",
                    "source_id": row.get("titulo", ""),
                    "titulo": row.get("titulo", ""),
                    "clip": row.get("clip", ""),
                    "training_usability": row.get("training_usability", ""),
                    "policy_moderate": row.get("policy_moderate", ""),
                }
            )
        return rows
    return leer_csv(Path(config["test_split"]))


def parse_experiment(config_path: Path, raw_dir: Path, results_dir: Path) -> dict[str, object]:
    config = leer_config(config_path)
    experiment = config["experiment"]
    raw_experiment_dir = raw_dir / experiment
    inf_path = raw_experiment_dir / "test.inf"
    if not inf_path.exists():
        return {"experiment": experiment, "status": "pending", "reason": f"falta {inf_path}"}

    test_rows = _metadata_rows(config, raw_experiment_dir)
    pairs = leer_inf(inf_path)
    if len(test_rows) != len(pairs):
        raise ValueError(f"{experiment}: test rows={len(test_rows)} pero inf lines={len(pairs)}")

    rows = []
    for meta, (reference, hypothesis) in zip(test_rows, pairs):
        row = {
            "experiment": experiment,
            "source_id": meta.get("source_id") or meta.get("titulo", ""),
            "clip": meta.get("clip", ""),
            "split": meta.get("split", "test"),
            "training_usability": meta.get("training_usability", ""),
            "policy_moderate": meta.get("policy_moderate", ""),
            "preprocessing_variant": config["preprocessing_variant"],
            "transcript_variant": config["transcript_variant"],
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": f"{wer(reference, hypothesis):.6f}",
            "cer": f"{cer(reference, hypothesis):.6f}",
        }
        rows.append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{experiment}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "experiment": experiment,
        "status": "parsed",
        "rows": len(rows),
        "output": str(out),
        "wer": mean(float(row["wer"]) for row in rows) if rows else None,
        "cer": mean(float(row["cer"]) for row in rows) if rows else None,
    }


def parse_all(output_base: Path = DEFAULT_OUTPUT_BASE) -> dict[str, object]:
    experiments_dir = output_base / "experiments"
    raw_dir = output_base / "raw"
    results_dir = output_base / "results"
    summaries = [
        parse_experiment(path, raw_dir, results_dir)
        for path in sorted(experiments_dir.glob("*/experiment_config.json"))
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["experiment", "status", "rows", "wer", "cer", "reason", "output"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    return {"summary": str(summary_path), "experiments": summaries}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(parse_all(args.output_base), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
