"""Persistencia JSONL de feedback revisable."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from segmentacion_oraciones.src.contracts import FeedbackEvent


VALID_USER_DECISIONS = {"accept", "edit", "reject", "unclear"}
VALID_REVIEW_STATUSES = {"pending", "valid", "discarded", "needs_review"}


class FeedbackWriter:
    """Escribe eventos de feedback como JSONL local."""

    def __init__(self, path: str | Path = "segmentacion_oraciones/outputs/feedback/events.jsonl") -> None:
        self.path = Path(path)

    def write(self, event: FeedbackEvent | dict[str, Any]) -> Path:
        data = validate_feedback_event(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
        return self.path


def validate_feedback_event(event: FeedbackEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, dict):
        data = dict(event)
    elif is_dataclass(event):
        data = asdict(event)
    elif hasattr(event, "to_dict"):
        data = event.to_dict()
    else:
        raise TypeError(f"Evento de feedback no soportado: {type(event)!r}")

    required = {
        "segment_id",
        "raw_vsr_text",
        "committed_text",
        "corrected_text",
        "user_decision",
        "review_status",
    }
    missing = sorted(name for name in required if name not in data)
    if missing:
        raise ValueError(f"Faltan campos de feedback: {', '.join(missing)}")
    if not str(data["segment_id"]).strip():
        raise ValueError("segment_id no puede estar vacio")
    if data["user_decision"] not in VALID_USER_DECISIONS:
        raise ValueError(f"user_decision invalido: {data['user_decision']!r}")
    if data["review_status"] not in VALID_REVIEW_STATUSES:
        raise ValueError(f"review_status invalido: {data['review_status']!r}")

    data.setdefault("metadata", {})
    data.setdefault("latency_ms", 0.0)
    data.setdefault("source", "")
    data.setdefault("split", "")
    data.setdefault("user_correction", "")
    return data
