"""Corrector conservador inicial."""

from __future__ import annotations

from realtime.src.contracts import CorrectionResult
from realtime.src.validation import fallback_correction, validate_correction_result


class IdentityCorrectionProvider:
    """Corrector no-op.

    Es la baseline segura: nunca inventa, nunca cambia significado y permite
    medir la latencia/logging del flujo antes de conectar un LLM real.
    """

    name = "identity_correction"

    def correct(self, raw_text: str, context_before: str = "", context_after: str = "") -> CorrectionResult:
        try:
            result = CorrectionResult(
                raw_text=raw_text,
                corrected_text=raw_text,
                confidence=1.0,
                changed=False,
                edits=[],
                risk_flags=[],
            )
        except Exception:
            return fallback_correction(raw_text, "identity_exception")
        validated, _ = validate_correction_result(result, raw_text)
        return validated
