"""Helpers de lectura para el notebook batch VSR."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"


def cargar_configs(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    rows = []
    for path in sorted((output_base / "experiments").glob("*/experiment_config.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def tamanos_experimentos(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, config in configs.iterrows():
        train = pd.read_csv(config["train_split"]) if Path(config["train_split"]).exists() else pd.DataFrame()
        val = pd.read_csv(config["val_split"]) if Path(config["val_split"]).exists() else pd.DataFrame()
        test = pd.read_csv(config["test_split"]) if Path(config["test_split"]).exists() else pd.DataFrame()
        rows.append(
            {
                "experiment": config["experiment"],
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
        )
    return pd.DataFrame(rows)


def transcript_changes(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "transcript_cleaning_changes.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def transcript_candidates(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "transcript_cleaning_candidates.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def transcript_policy(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "transcript_quality_policy.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def transcript_asr2(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "transcript_second_pass_asr.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def transcript_disagreement(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "transcript_asr_disagreement.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def transcript_summary(
    changes: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    policy: pd.DataFrame | None = None,
    asr2: pd.DataFrame | None = None,
    disagreement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame([{"transcripts": 0, "changed": 0, "unchanged": 0, "decision": "BLOCKED_MISSING_ASR2"}])
    candidates = candidates if candidates is not None else pd.DataFrame()
    policy = policy if policy is not None else pd.DataFrame()
    asr2 = asr2 if asr2 is not None else pd.DataFrame()
    disagreement = disagreement if disagreement is not None else pd.DataFrame()
    changed = int((changes["changed"].astype(str).str.lower() == "true").sum())
    bad = int(policy.get("transcript_usability", pd.Series(dtype=str)).eq("bad_candidate").sum()) if not policy.empty else 0
    questionable = int(policy.get("transcript_usability", pd.Series(dtype=str)).eq("questionable").sum()) if not policy.empty else 0
    asr_ok = int(asr2.get("status", pd.Series(dtype=str)).eq("ok").sum()) if not asr2.empty else 0
    high = int(disagreement.get("disagreement_level", pd.Series(dtype=str)).eq("high").sum()) if not disagreement.empty else 0
    auto_replacements = int(candidates.get("auto_applied", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum()) if not candidates.empty else 0
    if asr_ok == 0:
        decision = "BLOCKED_MISSING_ASR2"
    elif bad or high or auto_replacements:
        decision = "READY_FOR_VM"
    elif changed <= 52 and len(candidates) <= 30:
        decision = "LOW_IMPACT_DO_NOT_PRIORITIZE"
    else:
        decision = "REVIEW_NEEDED"
    return pd.DataFrame(
        [
            {
                "transcripts": len(changes),
                "changed": changed,
                "unchanged": len(changes) - changed,
                "asr2_ok": asr_ok,
                "disagreement_high": high,
                "replacement_candidates": len(candidates),
                "auto_replacements": auto_replacements,
                "questionable": questionable,
                "bad_candidate": bad,
                "decision": decision,
            }
        ]
    )


def preprocessing_smoke(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "preprocessing_variant_manifest_smoke.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def resultados(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "results" / "summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
