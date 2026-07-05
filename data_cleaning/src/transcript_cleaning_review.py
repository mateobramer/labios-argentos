"""Tablas para revisar transcript cleaning stronger con ASR2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def cargar_outputs(output_base: Path = OUTPUT_BASE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    changes = _read_optional(output_base / "transcript_cleaning_changes.csv")
    candidates = _read_optional(output_base / "transcript_cleaning_candidates.csv")
    policy = _read_optional(output_base / "transcript_quality_policy.csv")
    asr2 = _read_optional(output_base / "transcript_second_pass_asr.csv")
    disagreement = _read_optional(output_base / "transcript_asr_disagreement.csv")
    return changes, candidates, policy, asr2, disagreement


def resumen(
    changes: pd.DataFrame,
    candidates: pd.DataFrame,
    policy: pd.DataFrame,
    asr2: pd.DataFrame,
    disagreement: pd.DataFrame,
) -> pd.DataFrame:
    changed = changes.get("changed", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
    auto = changes.get("auto_applied", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
    status_counts = asr2.get("status", pd.Series(dtype=str)).astype(str).value_counts()
    level_counts = disagreement.get("disagreement_level", pd.Series(dtype=str)).astype(str).value_counts()
    return pd.DataFrame(
        [
            {
                "transcripts": len(changes),
                "auto_clean_changes": int((changed & auto).sum()) if len(changes) else 0,
                "asr2_rows": len(asr2),
                "asr2_ok": int(status_counts.get("ok", 0)),
                "asr2_blocked": int(status_counts.get("blocked", 0)),
                "disagreement_rows": len(disagreement),
                "disagreement_high": int(level_counts.get("high", 0)),
                "replacement_candidates": len(candidates),
                "auto_replacements": int(candidates.get("auto_applied", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum()),
                "usable": int(policy.get("transcript_usability", pd.Series(dtype=str)).eq("usable").sum()),
                "questionable": int(policy.get("transcript_usability", pd.Series(dtype=str)).eq("questionable").sum()),
                "bad_candidate": int(policy.get("transcript_usability", pd.Series(dtype=str)).eq("bad_candidate").sum()),
                "excluded_train": excluded_train(policy),
            }
        ]
    )


def cambios_por_tipo(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame(columns=["change_type", "clips"])
    changed = changes[changes["changed"].astype(str).str.lower() == "true"]
    if changed.empty:
        return pd.DataFrame(columns=["change_type", "clips"])
    return changed.groupby("change_type").size().rename("clips").reset_index().sort_values("clips", ascending=False)


def usability_counts(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame(columns=["transcript_usability", "clips"])
    return policy.groupby("transcript_usability").size().rename("clips").reset_index().sort_values("clips", ascending=False)


def disagreement_counts(disagreement: pd.DataFrame) -> pd.DataFrame:
    if disagreement.empty:
        return pd.DataFrame(columns=["disagreement_level", "clips"])
    return disagreement.groupby("disagreement_level").size().rename("clips").reset_index().sort_values("clips", ascending=False)


def ejemplos_autoaplicados(changes: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame()
    mask = changes["changed"].astype(str).str.lower().eq("true")
    cols = [c for c in ["source_id", "clip", "change_type", "evidence", "original_text", "asr2_text", "cleaned_text"] if c in changes.columns]
    return changes.loc[mask, cols].head(n)


def candidatos_por_tipo(candidates: pd.DataFrame, candidate_type: str, n: int = 10) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    cols = [c for c in ["source_id", "clip", "candidate_type", "span", "suggestion", "evidence", "confidence", "current_text", "asr2_text", "reason_not_auto_applied"] if c in candidates.columns]
    return candidates.loc[candidates["candidate_type"].eq(candidate_type), cols].head(n)


def candidatos_no_autoaplicados(candidates: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    mask = candidates["auto_applied"].astype(str).str.lower().eq("false")
    cols = [c for c in ["source_id", "clip", "candidate_type", "span", "suggestion", "evidence", "confidence", "reason_not_auto_applied"] if c in candidates.columns]
    return candidates.loc[mask, cols].head(n)


def top_fuentes_problemas(policy: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame(columns=["source_id", "problem_clips"])
    badish = policy[policy["transcript_usability"].isin(["questionable", "bad_candidate"])]
    if badish.empty:
        return pd.DataFrame(columns=["source_id", "problem_clips"])
    return badish.groupby("source_id").size().rename("problem_clips").reset_index().sort_values("problem_clips", ascending=False).head(n)


def impacto_train(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame([{"train_transcripts": 0, "excluded_bad_candidate": 0, "kept": 0, "excluded_pct": 0.0}])
    train = policy[policy["split"] == "train"]
    excluded = train["transcript_policy_moderate"].eq("exclude").sum()
    return pd.DataFrame(
        [
            {
                "train_transcripts": len(train),
                "excluded_bad_candidate": int(excluded),
                "kept": int(len(train) - excluded),
                "excluded_pct": round((excluded / len(train)) * 100, 2) if len(train) else 0.0,
            }
        ]
    )


def excluded_train(policy: pd.DataFrame) -> int:
    if policy.empty:
        return 0
    train = policy[policy["split"] == "train"]
    return int(train["transcript_policy_moderate"].eq("exclude").sum())


def decision(changes: pd.DataFrame, candidates: pd.DataFrame, policy: pd.DataFrame, asr2: pd.DataFrame, disagreement: pd.DataFrame) -> str:
    if asr2.empty:
        return "BLOCKED_MISSING_ASR2: text-only audit cannot detect hallucinations/misalignment reliably."
    if not asr2["status"].astype(str).eq("ok").any():
        return "BLOCKED_MISSING_ASR2: text-only audit cannot detect hallucinations/misalignment reliably."
    bad = int(policy.get("transcript_usability", pd.Series(dtype=str)).eq("bad_candidate").sum())
    high = int(disagreement.get("disagreement_level", pd.Series(dtype=str)).eq("high").sum())
    auto = int(candidates.get("auto_applied", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum())
    if bad or high or auto:
        return "READY_FOR_VM"
    changed = int(changes.get("changed", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum())
    if changed <= 52 and len(candidates) <= 30:
        return "WEAK_NO_IMPACT"
    return "REVIEW_NEEDED"
