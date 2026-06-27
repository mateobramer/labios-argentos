"""Logging JSONL tolerante a fallos para debugging y evaluacion."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class JSONLLogger:
    """Logger simple que nunca debe romper el flujo principal."""

    def __init__(self, path: str | Path = "segmentacion_oraciones/outputs/llm_logs/events.jsonl") -> None:
        self.path = Path(path)

    def write(self, event: dict[str, Any]) -> tuple[bool, float, str]:
        started = time.perf_counter()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = _jsonable(event)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            return True, _elapsed_ms(started), ""
        except Exception as exc:
            return False, _elapsed_ms(started), str(exc)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def provider_log_event(
    *,
    provider: str,
    stage: str,
    input_payload: Any,
    output_payload: Any,
    latency_ms: float,
    valid: bool,
    fallback_used: bool,
    error: str = "",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "stage": stage,
        "input": input_payload,
        "output": output_payload,
        "latency_ms": latency_ms,
        "valid": valid,
        "fallback_used": fallback_used,
        "error": error,
    }
