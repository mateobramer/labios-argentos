"""Tablas para revisar transcript cleaning stronger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"


def cargar_outputs(output_base: Path = OUTPUT_BASE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    changes = pd.read_csv(output_base / "transcript_cleaning_changes.csv")
    candidates = pd.read_csv(output_base / "transcript_cleaning_candidates.csv")
    policy = pd.read_csv(output_base / "transcript_quality_policy.csv")
    return changes, candidates, policy


def resumen(changes: pd.DataFrame, candidates: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    changed = changes["changed"].astype(str).str.lower() == "true"
    auto = changes["auto_applied"].astype(str).str.lower() == "true"
    excluded = policy["transcript_policy_moderate"].eq("exclude").sum()
    return pd.DataFrame(
        [
            {
                "transcripts": len(changes),
                "auto_clean_safe_changed": int((changed & auto).sum()),
                "aggressive_candidates": len(candidates),
                "usable": int(policy["transcript_usability"].eq("usable").sum()),
                "questionable": int(policy["transcript_usability"].eq("questionable").sum()),
                "bad_candidate": int(policy["transcript_usability"].eq("bad_candidate").sum()),
                "excluded_policy_moderate": int(excluded),
            }
        ]
    )


def cambios_por_tipo(changes: pd.DataFrame) -> pd.DataFrame:
    changed = changes[changes["changed"].astype(str).str.lower() == "true"]
    if changed.empty:
        return pd.DataFrame(columns=["change_type", "clips"])
    return changed.groupby("change_type").size().rename("clips").reset_index().sort_values("clips", ascending=False)


def usability_counts(policy: pd.DataFrame) -> pd.DataFrame:
    return policy.groupby("transcript_usability").size().rename("clips").reset_index().sort_values("clips", ascending=False)


def ejemplos_autoaplicados(changes: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    mask = changes["changed"].astype(str).str.lower().eq("true")
    cols = ["source_id", "clip", "change_type", "evidence", "original_text", "cleaned_text"]
    return changes.loc[mask, cols].head(n)


def candidatos_no_autoaplicados(candidates: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    mask = candidates["auto_applied"].astype(str).str.lower().eq("false")
    cols = [
        "source_id",
        "clip",
        "candidate_type",
        "span",
        "suggestion",
        "evidence",
        "confidence",
        "reason_not_auto_applied",
    ]
    return candidates.loc[mask, cols].head(n)


def top_fuentes_cambios(changes: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    changed = changes[changes["changed"].astype(str).str.lower() == "true"]
    if changed.empty:
        return pd.DataFrame(columns=["source_id", "changed"])
    return changed.groupby("source_id").size().rename("changed").reset_index().sort_values("changed", ascending=False).head(n)


def top_fuentes_bad(policy: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    bad = policy[policy["transcript_usability"] == "bad_candidate"]
    if bad.empty:
        return pd.DataFrame(columns=["source_id", "bad_candidate"])
    return bad.groupby("source_id").size().rename("bad_candidate").reset_index().sort_values("bad_candidate", ascending=False).head(n)


def impacto_train(policy: pd.DataFrame) -> pd.DataFrame:
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


def decision(changes: pd.DataFrame, candidates: pd.DataFrame, policy: pd.DataFrame) -> str:
    bad = int(policy["transcript_usability"].eq("bad_candidate").sum())
    cand = len(candidates)
    if bad or cand:
        return "revisar: hay candidatos agresivos y/o bad_candidate antes de correr VM."
    if changes["changed"].astype(str).str.lower().eq("true").sum() == 0:
        return "listo tecnicamente, pero el efecto esperado es bajo porque no hay cambios."
    return "listo para revision manual: solo auto_clean_safe y sin bloqueos."
