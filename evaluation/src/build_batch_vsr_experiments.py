"""Genera configs E0-E4 para la corrida batch VSR futura."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
VISUAL_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "visual_cleaning"


def _exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def leer_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def transcript_policy_summary(output_base: Path) -> dict[str, object]:
    rows = leer_csv(output_base / "transcript_quality_policy.csv")
    counts = Counter(row.get("transcript_usability", "") for row in rows)
    candidates = leer_csv(output_base / "transcript_cleaning_candidates.csv")
    asr2 = leer_csv(output_base / "transcript_second_pass_asr.csv")
    disagreement = leer_csv(output_base / "transcript_asr_disagreement.csv")
    asr2_counts = Counter(row.get("status", "") for row in asr2)
    disagreement_counts = Counter(row.get("disagreement_level", "") for row in disagreement)
    auto_replacements = sum(1 for row in candidates if str(row.get("auto_applied", "")).lower() == "true")
    if not asr2 or asr2_counts.get("ok", 0) == 0:
        decision = "BLOCKED_MISSING_ASR2"
    elif counts.get("bad_candidate", 0) or disagreement_counts.get("high", 0) or auto_replacements:
        decision = "READY_FOR_VM"
    elif len(candidates) <= 30 and counts.get("bad_candidate", 0) == 0:
        decision = "LOW_IMPACT_DO_NOT_PRIORITIZE"
    else:
        decision = "REVIEW_NEEDED"
    return {
        "transcript_excluded_count": counts.get("bad_candidate", 0),
        "transcript_usability_counts": dict(counts),
        "replacement_candidates": len(candidates),
        "auto_replacements": auto_replacements,
        "asr2_rows": len(asr2),
        "asr2_counts": dict(asr2_counts),
        "disagreement_rows": len(disagreement),
        "disagreement_counts": dict(disagreement_counts),
        "transcript_decision": decision,
    }


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


def _status_for_transcript_experiment(
    train_split: Path,
    val_split: Path,
    test_split: Path,
    root: Path,
    transcript_decision: str,
) -> tuple[str, str]:
    status, reason = _status_for_transcripts(train_split, val_split, test_split, root)
    if status == "blocked":
        return status, reason
    if transcript_decision == "READY_FOR_VM":
        return "ready", ""
    if transcript_decision == "BLOCKED_MISSING_ASR2":
        return "blocked_missing_asr2", "falta ASR2 usable; no entrenar E2 como mejora fuerte"
    if transcript_decision == "LOW_IMPACT_DO_NOT_PRIORITIZE":
        return "low_impact_do_not_prioritize", "ASR2 disponible pero el transcript audit no cambia train de forma util"
    return "review_needed", "revisar notebook 08 antes de entrenar E2"


def _variant_manifest_status(rois_root: Path) -> tuple[bool, str]:
    manifest = rois_root.parent / "preprocessing_variant_manifest_full.csv"
    if not manifest.exists():
        return False, f"generar manifest full lower_face_resized96: {manifest}"
    rows = leer_csv(manifest)
    expected = leer_csv(REPO_ROOT / "vsr_models" / "splits" / "splits.csv")
    if len(rows) != len(expected):
        return False, f"manifest full incompleto: rows={len(rows)} expected={len(expected)}"
    for row in rows:
        if row.get("status") != "ok":
            return False, "manifest full tiene status no-ok"
        if row.get("dtype") != "uint8":
            return False, "manifest full tiene dtype no uint8"
        if not re.match(r"^\d+x96x96$", str(row.get("shape", ""))):
            return False, "manifest full tiene shape no compatible con T x 96 x 96"
        if not Path(row.get("variant_roi_path", "")).exists():
            return False, "manifest full apunta a ROI faltante"
    return True, ""


def _status_for_variant(train_split: Path, val_split: Path, test_split: Path, rois_root: Path) -> tuple[str, str]:
    status, reason = _status_for_current(train_split, val_split, test_split)
    if status == "blocked":
        return status, reason
    ready, manifest_reason = _variant_manifest_status(rois_root)
    if ready:
        return "ready", ""
    return "ready_after_generation", manifest_reason or f"generar ROIs lower_face_resized96 en VM: {rois_root}"


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
    transcript_policy: str = "none",
    transcript_excluded_count: int = 0,
    transcript_usability_counts: dict[str, int] | None = None,
    transcript_decision: str = "",
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
        "transcript_policy": transcript_policy,
        "transcript_excluded_count": transcript_excluded_count,
        "transcript_usability_counts": transcript_usability_counts or {},
        "transcript_decision": transcript_decision,
        "preprocessing_variant": preprocessing_variant,
        "blocked_reason": blocked_reason,
    }


def build_configs(output_base: Path = DEFAULT_OUTPUT_BASE) -> dict[str, dict[str, object]]:
    original_train = VISUAL_OUTPUT_BASE / "splits_original" / "train.csv"
    original_val = VISUAL_OUTPUT_BASE / "splits_original" / "val.csv"
    cleaned_train = VISUAL_OUTPUT_BASE / "splits_visual_cleaned" / "train.csv"
    cleaned_val = VISUAL_OUTPUT_BASE / "splits_visual_cleaned" / "val.csv"
    transcript_train = output_base / "splits_transcript_cleaned_stronger" / "train.csv"
    transcript_val = output_base / "splits_transcript_cleaned_stronger" / "val.csv"
    combined_train = output_base / "splits_all_combined" / "train.csv"
    combined_val = output_base / "splits_all_combined" / "val.csv"
    test_original = VISUAL_OUTPUT_BASE / "manifests" / "original_test.csv"
    current_rois = REPO_ROOT / "data" / "processed" / "lip_rois"
    cleaned_transcripts = output_base / "transcripts_cleaned_stronger"
    variant_rois = output_base / "rois_lower_face_resized96"
    policy = transcript_policy_summary(output_base)

    e0_status, e0_reason = _status_for_current(original_train, original_val, test_original)
    e1_status, e1_reason = _status_for_current(cleaned_train, cleaned_val, test_original)
    e2_status, e2_reason = _status_for_transcript_experiment(
        transcript_train,
        transcript_val,
        test_original,
        cleaned_transcripts,
        str(policy["transcript_decision"]),
    )
    e3_status, e3_reason = _status_for_variant(original_train, original_val, test_original, variant_rois)
    e4_uses_transcripts = policy["transcript_decision"] == "READY_FOR_VM"
    e4_train = combined_train if e4_uses_transcripts else cleaned_train
    e4_val = combined_val if e4_uses_transcripts else cleaned_val
    e4_transcripts = cleaned_transcripts if e4_uses_transcripts else None
    e4_transcript_variant = "transcript_cleaned_stronger" if e4_uses_transcripts else "current"
    e4_variant_status, e4_variant_reason = _status_for_variant(e4_train, e4_val, test_original, variant_rois)
    if e4_variant_status != "ready":
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
        "E2_transcript_cleaned_stronger": _config(
            "E2_transcript_cleaned_stronger",
            e2_status,
            transcript_train,
            transcript_val,
            test_original,
            current_rois,
            cleaned_transcripts,
            "none",
            "transcript_cleaned_stronger",
            "current",
            e2_reason,
            "moderate",
            int(policy["transcript_excluded_count"]),
            policy["transcript_usability_counts"],
            str(policy["transcript_decision"]),
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
            e4_train,
            e4_val,
            test_original,
            variant_rois,
            e4_transcripts,
            "conservative",
            e4_transcript_variant,
            "lower_face_resized96",
            e4_reason,
            "moderate" if e4_uses_transcripts else "none",
            int(policy["transcript_excluded_count"]),
            policy["transcript_usability_counts"],
            str(policy["transcript_decision"]),
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
