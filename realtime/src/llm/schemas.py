"""Schemas JSON para providers LLM con salida estricta."""

from __future__ import annotations


CLOSURE_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "committed_text", "confidence", "reason", "risk_flags"],
    "properties": {
        "action": {"type": "string", "enum": ["commit", "wait", "low_confidence"]},
        "committed_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


CORRECTION_RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["corrected_text", "confidence", "changed", "edits", "risk_flags"],
    "properties": {
        "corrected_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "changed": {"type": "boolean"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["from", "to", "type"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "type": {"type": "string"},
                },
            },
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}
