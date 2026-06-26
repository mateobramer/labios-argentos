"""Contratos estables para cierre, correccion y feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CommitAction(str, Enum):
    """Acciones posibles del modulo de cierre."""

    COMMIT = "commit"
    WAIT = "wait"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(slots=True)
class PartialHypothesis:
    """Texto parcial producido por VSR o por una simulacion offline."""

    partial_text: str
    last_tokens: list[str] = field(default_factory=list)
    stable_prefix: str = ""
    vsr_confidence: float | None = None
    ms_since_last_commit: int | None = None
    segment_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommitDecision:
    """Decision validada del cierre de oracion."""

    action: CommitAction
    committed_text: str = ""
    confidence: float = 0.0
    reason: str = ""
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


@dataclass(slots=True)
class CorrectionResult:
    """Resultado del corrector sobre texto ya commiteado."""

    raw_text: str
    corrected_text: str
    confidence: float = 0.0
    changed: bool = False
    edits: list[dict[str, Any]] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeedbackEvent:
    """Evento auditable de feedback humano o de simulacion."""

    segment_id: str
    raw_vsr_text: str
    committed_text: str
    corrected_text: str
    user_correction: str = ""
    user_decision: str = "unclear"
    latency_ms: float = 0.0
    source: str = ""
    split: str = ""
    review_status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderResult:
    """Wrapper comun para registrar proveedores reemplazables."""

    provider: str
    value: Any
    latency_ms: float = 0.0
    raw_output: Any = None
    fallback_used: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        value = data.get("value")
        if hasattr(value, "to_dict"):
            data["value"] = value.to_dict()
        return data
