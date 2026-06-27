"""Provider opcional para Ollama.

No se importa ni se ejecuta automaticamente. Usa solo stdlib y falla de forma
segura: cierre -> wait, correccion -> texto crudo.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from segmentacion_oraciones.src.contracts import CommitDecision, CorrectionResult, PartialHypothesis
from segmentacion_oraciones.src.llm.prompts import build_closure_prompt, build_correction_prompt
from segmentacion_oraciones.src.llm.schemas import CLOSURE_DECISION_SCHEMA, CORRECTION_RESULT_SCHEMA
from segmentacion_oraciones.src.validation import (
    fallback_commit,
    fallback_correction,
    validate_commit_decision,
    validate_correction_result,
)


class OllamaProvider:
    """Provider LLM local opcional via Ollama `/api/generate`."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 2.5,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.name = f"ollama:{model}"
        self.last_latency_ms = 0.0
        self.last_error = ""
        self.last_raw_output: Any = None

    def decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        started = time.perf_counter()
        try:
            raw = self._generate(build_closure_prompt(hypothesis), CLOSURE_DECISION_SCHEMA)
            self.last_raw_output = raw
            parsed = _parse_json_object(raw)
            decision, used_fallback = validate_commit_decision(parsed)
            if used_fallback:
                decision.risk_flags.append("ollama_invalid_output")
            return decision
        except Exception as exc:
            self.last_error = str(exc)
            return fallback_commit("ollama_unavailable_or_invalid")
        finally:
            self.last_latency_ms = _elapsed_ms(started)

    def correct(self, raw_text: str, context_before: str = "", context_after: str = "") -> CorrectionResult:
        started = time.perf_counter()
        try:
            raw = self._generate(
                build_correction_prompt(raw_text, context_before, context_after),
                CORRECTION_RESULT_SCHEMA,
            )
            self.last_raw_output = raw
            parsed = _parse_json_object(raw)
            result, used_fallback = validate_correction_result(parsed, raw_text)
            if used_fallback:
                result.risk_flags.append("ollama_invalid_output")
            return result
        except Exception as exc:
            self.last_error = str(exc)
            return fallback_correction(raw_text, "ollama_unavailable_or_invalid")
        finally:
            self.last_latency_ms = _elapsed_ms(started)

    def _generate(self, prompt: str, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_predict": 160,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama no disponible: {exc}") from exc

        data = json.loads(body)
        generated = data.get("response")
        if not isinstance(generated, str):
            raise ValueError("Respuesta de Ollama sin campo 'response' string")
        return generated


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("El LLM no devolvio un objeto JSON")
    return parsed


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
