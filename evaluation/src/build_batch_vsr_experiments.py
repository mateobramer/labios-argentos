"""Genera configs E0-E4 para la corrida batch VSR futura."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
VISUAL_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "visual_cleaning"


def _exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def _status_for_current(train_split: Path, val_split: Path, test_split: Path) -> tuple[str, str]:
    missing = [str(p) for p in (train_split, val_split, test_split) if not p.exists()]
    if missing:
        return "blocked", "faltan splits: " + ";".join(missing)
    return "ready", ""


def _status_for_transcripts(train_split: Path, val_split: Path, test_split: Path, root: Path) -> tuple[str, str]:
    status, reason = _status_for_current(train_split, val_split, test_split)
    if status == "blocked":
        return status, reason
    if not root.exists():
        return "blocked", f"faltan transcripts cleaned: {root}"
    return "ready", ""


def _status_for_variant(train_split: Path, val_split: Path, test_split: Path, rois_root: Path) -> tuple[str, str]:
    status, reason = _status_for_current(train_split, val_split, test_split)
    if status == "blocked":
        return status, reason
    if rois_root.exists() and any(rois_root.rglob("*.npz")):
        return "ready", ""
    return "ready_after_generation", f"generar ROIs lower_face_resized96 en VM: {rois_root}"


def _config(
    experiment: str,
    status: str,
    train_split: Path,
    val_split: Path,
    test_split: Path,
    rois_root: Path,
    transcripts_root: Path | None,
    visual_cleaning: str,
    transcript_variant: str,
    preprocessing_variant: str,
    blocked_reason: str,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "status": status,
        "train_split": str(train_split),
        "val_split": str(val_split),
        "test_split": str(test_split),
        "rois_root": str(rois_root),
        "transcripts_root": str(transcripts_root) if transcripts_root else "",
        "visual_cleaning": visual_cleaning,
        "transcript_variant": transcript_variant,
        "preprocessing_variant": preprocessing_variant,
        "blocked_reason": blocked_reason,
    }


def build_configs(output_base: Path = DEFAULT_OUTPUT_BASE) -> dict[str, dict[str, object]]:
    original_train = VISUAL_OUTPUT_BASE / "splits_original" / "train.csv"
    original_val = VISUAL_OUTPUT_BASE / "splits_original" / "val.csv"
    cleaned_train = VISUAL_OUTPUT_BASE / "splits_visual_cleaned" / "train.csv"
    cleaned_val = VISUAL_OUTPUT_BASE / "splits_visual_cleaned" / "val.csv"
    test_original = VISUAL_OUTPUT_BASE / "manifests" / "original_test.csv"
    current_rois = REPO_ROOT / "data" / "processed" / "lip_rois"
    cleaned_transcripts = output_base / "transcripts_cleaned_restricted"
    variant_rois = output_base / "rois_lower_face_resized96"

    e0_status, e0_reason = _status_for_current(original_train, original_val, test_original)
    e1_status, e1_reason = _status_for_current(cleaned_train, cleaned_val, test_original)
    e2_status, e2_reason = _status_for_transcripts(original_train, original_val, test_original, cleaned_transcripts)
    e3_status, e3_reason = _status_for_variant(original_train, original_val, test_original, variant_rois)
    e4_base_status, e4_base_reason = _status_for_transcripts(cleaned_train, cleaned_val, test_original, cleaned_transcripts)
    e4_variant_status, e4_variant_reason = _status_for_variant(cleaned_train, cleaned_val, test_original, variant_rois)
    if e4_base_status == "blocked":
        e4_status, e4_reason = e4_base_status, e4_base_reason
    elif e4_variant_status != "ready":
        e4_status, e4_reason = e4_variant_status, e4_variant_reason
    else:
        e4_status, e4_reason = "ready", ""

    configs = {
        "E0_baseline_original": _config(
            "E0_baseline_original",
            e0_status,
            original_train,
            original_val,
            test_original,
            current_rois,
            None,
            "none",
            "current",
            "current",
            e0_reason,
        ),
        "E1_visual_cleaned": _config(
            "E1_visual_cleaned",
            e1_status,
            cleaned_train,
            cleaned_val,
            test_original,
            current_rois,
            None,
            "conservative",
            "current",
            "current",
            e1_reason,
        ),
        "E2_transcript_cleaned": _config(
            "E2_transcript_cleaned",
            e2_status,
            original_train,
            original_val,
            test_original,
            current_rois,
            cleaned_transcripts,
            "none",
            "transcript_cleaned_restricted",
            "current",
            e2_reason,
        ),
        "E3_preprocessing_variant": _config(
            "E3_preprocessing_variant",
            e3_status,
            original_train,
            original_val,
            test_original,
            variant_rois,
            None,
            "none",
            "current",
            "lower_face_resized96",
            e3_reason,
        ),
        "E4_all_combined": _config(
            "E4_all_combined",
            e4_status,
            cleaned_train,
            cleaned_val,
            test_original,
            variant_rois,
            cleaned_transcripts,
            "conservative",
            "transcript_cleaned_restricted",
            "lower_face_resized96",
            e4_reason,
        ),
    }
    return configs


def write_configs(output_base: Path = DEFAULT_OUTPUT_BASE) -> dict[str, object]:
    configs = build_configs(output_base)
    experiments_dir = output_base / "experiments"
    for name, config in configs.items():
        out = experiments_dir / name / "experiment_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "output_base": str(output_base),
        "experiments": {name: config["status"] for name, config in configs.items()},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(write_configs(args.output_base), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
