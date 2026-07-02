"""Genera manifests para comparar splits originales vs cleaning visual."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr_models" / "splits" / "splits.csv"
DEFAULT_POLICY = REPO_ROOT / "data" / "metadata" / "visual_quality_policy_analysis_v2.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "outputs" / "visual_cleaning" / "manifests"

QUALITY_COLUMNS = [
    "source_id",
    "training_usability",
    "policy_moderate",
    "review_score",
    "quality_score",
    "visual_quality_label",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _policy_moderate(row: dict[str, str]) -> str:
    return row.get("policy_moderate", "") or row.get("policy_moderate_v2", "")


def _visual_quality_label(row: dict[str, str]) -> str:
    if row.get("visual_quality_label"):
        return row["visual_quality_label"]
    if row.get("training_usability"):
        return row["training_usability"]
    return row.get("quality_bucket", "")


def _indice_politica(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indice: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        clip = row.get("clip", "")
        for source_col in ("titulo", "source_id"):
            source = row.get(source_col, "")
            if source and clip:
                indice[(source, clip)] = row
    return indice


def enriquecer_splits(
    split_rows: list[dict[str, str]],
    policy_rows: list[dict[str, str]],
    allow_missing_policy: bool = False,
) -> tuple[list[dict[str, str]], int]:
    indice = _indice_politica(policy_rows)
    enriquecidos: list[dict[str, str]] = []
    missing = 0

    for row in split_rows:
        policy = indice.get((row.get("titulo", ""), row.get("clip", "")))
        nuevo = dict(row)
        if policy is None:
            missing += 1
            for col in QUALITY_COLUMNS:
                nuevo.setdefault(col, "")
        else:
            nuevo["source_id"] = policy.get("source_id") or policy.get("titulo") or row.get("titulo", "")
            nuevo["training_usability"] = policy.get("training_usability", "")
            nuevo["policy_moderate"] = _policy_moderate(policy)
            nuevo["review_score"] = policy.get("review_score", "")
            nuevo["quality_score"] = policy.get("quality_score", "")
            nuevo["visual_quality_label"] = _visual_quality_label(policy)
        enriquecidos.append(nuevo)

    if missing and not allow_missing_policy:
        raise ValueError(
            f"{missing} clips de splits.csv no tienen fila en el manifest visual. "
            "Usar --allow-missing-policy solo para diagnostico."
        )
    return enriquecidos, missing


def _fieldnames(split_fieldnames: list[str]) -> list[str]:
    salida = list(split_fieldnames)
    for col in QUALITY_COLUMNS:
        if col not in salida:
            salida.append(col)
    return salida


def construir_manifests(
    splits_path: Path = DEFAULT_SPLITS,
    policy_path: Path = DEFAULT_POLICY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    allow_missing_policy: bool = False,
) -> dict[str, object]:
    split_rows = leer_csv(splits_path)
    policy_rows = leer_csv(policy_path)
    if not split_rows:
        raise ValueError(f"Sin filas en {splits_path}")

    fieldnames = _fieldnames(list(split_rows[0].keys()))
    rows, missing = enriquecer_splits(split_rows, policy_rows, allow_missing_policy)

    por_split = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("train", "val", "test")
    }
    visual_cleaned_train = [
        row for row in por_split["train"] if row.get("training_usability") != "bad_candidate"
    ]
    visual_cleaned_val = list(por_split["val"])
    visual_cleaned_test_original = list(por_split["test"])

    outputs = {
        "original_train.csv": por_split["train"],
        "original_val.csv": por_split["val"],
        "original_test.csv": por_split["test"],
        "visual_cleaned_train.csv": visual_cleaned_train,
        "visual_cleaned_val.csv": visual_cleaned_val,
        "visual_cleaned_test_original.csv": visual_cleaned_test_original,
    }
    for nombre, filas in outputs.items():
        escribir_csv(output_dir / nombre, filas, fieldnames)

    excluded_train = len(por_split["train"]) - len(visual_cleaned_train)
    return {
        "splits_path": str(splits_path),
        "policy_path": str(policy_path),
        "output_dir": str(output_dir),
        "original_train": len(por_split["train"]),
        "original_val": len(por_split["val"]),
        "original_test": len(por_split["test"]),
        "visual_cleaned_train": len(visual_cleaned_train),
        "visual_cleaned_val": len(visual_cleaned_val),
        "visual_cleaned_test_original": len(visual_cleaned_test_original),
        "excluded_train_bad_candidate": excluded_train,
        "missing_policy_rows": missing,
        "validation_policy": "val original completo; test original completo",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--allow-missing-policy", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    resumen = construir_manifests(
        splits_path=args.splits,
        policy_path=args.policy,
        output_dir=args.output_dir,
        allow_missing_policy=args.allow_missing_policy,
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
