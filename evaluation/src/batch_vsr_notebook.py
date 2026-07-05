"""Helpers de lectura para el notebook batch VSR."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
REPO_ANCHORS = (
    "evaluation/",
    "data/",
    "data_cleaning/",
    "visual_preprocessing/",
    "vsr_models/",
)


def resolver_repo_path(path_value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Mapea paths absolutos de VM al checkout local cuando es posible."""
    raw = str(path_value)
    path = Path(raw)
    if path.exists():
        return path
    normalized = raw.replace("\\", "/")
    for anchor in REPO_ANCHORS:
        if normalized.startswith(anchor):
            return repo_root / normalized
        marker = f"/{anchor}"
        if marker in normalized:
            return repo_root / anchor / normalized.split(marker, 1)[1]
    return path


def cargar_configs(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    rows = []
    for path in sorted((output_base / "experiments").glob("*/experiment_config.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def tamanos_experimentos(configs: pd.DataFrame, repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    rows = []
    for _, config in configs.iterrows():
        train_path = resolver_repo_path(config["train_split"], repo_root)
        val_path = resolver_repo_path(config["val_split"], repo_root)
        test_path = resolver_repo_path(config["test_split"], repo_root)
        train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
        val = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame()
        test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
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


def comparar_resultados(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    parsed = results[results["status"].eq("parsed")].copy()
    if parsed.empty:
        return parsed
    parsed["wer"] = pd.to_numeric(parsed["wer"], errors="coerce")
    parsed["cer"] = pd.to_numeric(parsed["cer"], errors="coerce")
    baseline = parsed[parsed["experiment"].eq("E0_baseline_original")]
    if baseline.empty:
        parsed["delta_wer_vs_e0"] = pd.NA
        parsed["delta_cer_vs_e0"] = pd.NA
    else:
        e0 = baseline.iloc[0]
        parsed["delta_wer_vs_e0"] = parsed["wer"] - e0["wer"]
        parsed["delta_cer_vs_e0"] = parsed["cer"] - e0["cer"]
    parsed["rank_wer"] = parsed["wer"].rank(method="min").astype("Int64")
    parsed["interpretacion"] = parsed["delta_wer_vs_e0"].map(
        lambda delta: "baseline"
        if pd.isna(delta) or abs(delta) < 1e-12
        else ("mejora_vs_e0" if delta < 0 else "empeora_vs_e0")
    )
    return parsed[
        [
            "experiment",
            "rows",
            "wer",
            "cer",
            "delta_wer_vs_e0",
            "delta_cer_vs_e0",
            "rank_wer",
            "interpretacion",
            "output",
        ]
    ].sort_values(["rank_wer", "experiment"])
