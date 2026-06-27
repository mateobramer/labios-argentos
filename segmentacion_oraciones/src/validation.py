"""Validacion estricta y fallbacks seguros para providers segmentacion_oraciones."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from segmentacion_oraciones.src.contracts import CommitAction, CommitDecision, CorrectionResult


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Salida no serializable como dict: {type(value)!r}")


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def fallback_commit(reason: str = "fallback_wait") -> CommitDecision:
    return CommitDecision(
        action=CommitAction.WAIT,
        committed_text="",
        confidence=0.0,
        reason=reason,
        risk_flags=["fallback"],
    )


def validate_commit_decision(value: Any) -> tuple[CommitDecision, bool]:
    """Devuelve una decision valida y si se uso fallback."""

    try:
        data = _as_dict(value)
        action = CommitAction(data.get("action"))
        committed_text = str(data.get("committed_text") or "").strip()
        confidence = _confidence(data.get("confidence", 0.0))
        reason = str(data.get("reason") or "").strip()
        flags_raw = data.get("risk_flags", [])
        risk_flags = [str(flag) for flag in flags_raw] if isinstance(flags_raw, list) else []

        if action == CommitAction.COMMIT and not committed_text:
            return fallback_commit("invalid_commit_without_text"), True
        if action != CommitAction.COMMIT:
            committed_text = ""
        if not reason:
            reason = "validated"

        return (
            CommitDecision(
                action=action,
                committed_text=committed_text,
                confidence=confidence,
                reason=reason,
                risk_flags=risk_flags,
            ),
            False,
        )
    except Exception:
        return fallback_commit("invalid_commit_decision"), True


def fallback_correction(raw_text: str, reason: str = "fallback_raw_text") -> CorrectionResult:
    return CorrectionResult(
        raw_text=raw_text,
        corrected_text=raw_text,
        confidence=0.0,
        changed=False,
        edits=[],
        risk_flags=["fallback", reason],
    )


def validate_correction_result(value: Any, raw_text: str) -> tuple[CorrectionResult, bool]:
    """Devuelve una correccion valida y si se uso fallback."""

    try:
        data = _as_dict(value)
        corrected_text = str(data.get("corrected_text") or "").strip()
        if not corrected_text:
            return fallback_correction(raw_text, "empty_corrected_text"), True

        # Control simple contra expansiones agresivas. El corrector puede mejorar forma,
        # pero no deberia convertir una frase corta en una explicacion nueva.
        raw_len = max(1, len(raw_text.strip()))
        if len(corrected_text) > raw_len * 2.5 and len(corrected_text) - raw_len > 30:
            return fallback_correction(raw_text, "overexpanded_correction"), True

        edits_raw = data.get("edits", [])
        edits = edits_raw if isinstance(edits_raw, list) else []
        flags_raw = data.get("risk_flags", [])
        risk_flags = [str(flag) for flag in flags_raw] if isinstance(flags_raw, list) else []
        changed = bool(data.get("changed", corrected_text != raw_text))

        return (
            CorrectionResult(
                raw_text=str(data.get("raw_text") or raw_text),
                corrected_text=corrected_text,
                confidence=_confidence(data.get("confidence", 0.0)),
                changed=changed,
                edits=edits,
                risk_flags=risk_flags,
            ),
            False,
        )
    except Exception:
        return fallback_correction(raw_text, "invalid_correction_result"), True
