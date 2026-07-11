"""Audita desacuerdos entre transcript actual y ASR2."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from vsr.evaluation.src.experiment_metrics import cer, wer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASR2 = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr" / "transcript_second_pass_asr.csv"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr" / "transcript_asr_disagreement.csv"

OUTPUT_COLUMNS = [
    "source_id",
    "clip",
    "split",
    "current_text",
    "asr2_text",
    "wer_current_vs_asr2",
    "cer_current_vs_asr2",
    "token_diff_summary",
    "disagreement_level",
    "reasons",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalizar_texto(texto: str) -> str:
    return " ".join(str(texto or "").strip().lower().split())


def _tokens(texto: str) -> list[str]:
    return [t for t in normalizar_texto(texto).split() if t]


def token_diff_summary(current_text: str, asr2_text: str, max_parts: int = 4) -> str:
    current = _tokens(current_text)
    asr2 = _tokens(asr2_text)
    matcher = SequenceMatcher(None, current, asr2)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = " ".join(current[i1:i2])[:60]
        right = " ".join(asr2[j1:j2])[:60]
        parts.append(f"{tag}: '{left}' -> '{right}'")
        if len(parts) >= max_parts:
            break
    return " | ".join(parts)


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def clasificar(current_text: str, asr2_text: str, status: str, reason: str) -> tuple[float, float, str, list[str]]:
    if status != "ok":
        return 1.0, 1.0, "blocked", ["asr2_blocked", reason or status]

    current = normalizar_texto(current_text)
    asr2 = normalizar_texto(asr2_text)
    current_tokens = _tokens(current)
    asr2_tokens = _tokens(asr2)
    w = wer(current, asr2)
    c = cer(current, asr2)
    reasons: list[str] = []

    if not current_tokens and asr2_tokens:
        reasons.extend(["possible_empty_or_no_speech", "possible_audio_text_mismatch"])
        return w, c, "high", reasons
    if current_tokens and not asr2_tokens:
        reasons.extend(["possible_empty_or_no_speech", "possible_whisper_hallucination"])
        return w, c, "high" if len(current_tokens) >= 4 else "medium", reasons
    if not current_tokens and not asr2_tokens:
        return w, c, "low", []

    overlap = len(set(current_tokens) & set(asr2_tokens)) / max(len(set(current_tokens) | set(asr2_tokens)), 1)
    length_ratio = max(len(current_tokens), len(asr2_tokens)) / max(min(len(current_tokens), len(asr2_tokens)), 1)
    if w >= 0.75 or c >= 0.55:
        level = "high"
    elif w >= 0.35 or c >= 0.25:
        level = "medium"
    else:
        level = "low"

    if level in {"medium", "high"}:
        if overlap <= 0.25 and len(current_tokens) >= 4 and len(asr2_tokens) >= 4:
            reasons.append("possible_timestamp_misalignment")
        if length_ratio >= 2.5:
            reasons.append("possible_audio_text_mismatch")
        if _repetition_ratio(asr2_tokens) >= 0.55 and len(asr2_tokens) >= 6:
            reasons.append("possible_whisper_hallucination")
        if _looks_like_entity_error(current_tokens, asr2_tokens):
            reasons.append("possible_entity_error")
    if not reasons and level != "low":
        reasons.append("possible_audio_text_mismatch" if level == "high" else "asr_disagreement")
    return w, c, level, reasons


def _looks_like_entity_error(current_tokens: list[str], asr2_tokens: list[str]) -> bool:
    if abs(len(current_tokens) - len(asr2_tokens)) > 4:
        return False
    matcher = SequenceMatcher(None, current_tokens, asr2_tokens)
    replaced = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
    return 1 <= replaced <= 4


def build_alignment_audit(asr2_path: Path = DEFAULT_ASR2, output_path: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    rows = leer_csv(asr2_path)
    out_rows: list[dict[str, str]] = []
    for row in rows:
        current_text = row.get("current_text", "")
        asr2_text = row.get("asr2_text", "")
        w, c, level, reasons = clasificar(current_text, asr2_text, row.get("status", ""), row.get("reason", ""))
        out_rows.append(
            {
                "source_id": row.get("source_id", ""),
                "clip": row.get("clip", ""),
                "split": row.get("split", ""),
                "current_text": current_text,
                "asr2_text": asr2_text,
                "wer_current_vs_asr2": f"{w:.6f}",
                "cer_current_vs_asr2": f"{c:.6f}",
                "token_diff_summary": token_diff_summary(current_text, asr2_text),
                "disagreement_level": level,
                "reasons": ";".join(r for r in reasons if r),
            }
        )
    escribir_csv(output_path, out_rows)
    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["disagreement_level"]] = counts.get(row["disagreement_level"], 0) + 1
    return {"rows": len(out_rows), "levels": counts, "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asr2", type=Path, default=DEFAULT_ASR2)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_alignment_audit(args.asr2, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
