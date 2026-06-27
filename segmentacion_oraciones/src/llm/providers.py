"""Interfaces reemplazables para cierre y correccion.

Este modulo no importa backends externos. Providers futuros de Ollama, OpenAI,
llama.cpp o PyTorch deben implementar estos protocolos sin volver obligatorias
esas dependencias para el MVP.
"""

from __future__ import annotations

from typing import Protocol

from segmentacion_oraciones.src.contracts import CommitDecision, CorrectionResult, PartialHypothesis


class ClosureProvider(Protocol):
    name: str

    def decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        """Decidir commit/wait/low_confidence sobre texto parcial."""


class CorrectionProvider(Protocol):
    name: str

    def correct(self, raw_text: str, context_before: str = "", context_after: str = "") -> CorrectionResult:
        """Corregir texto ya commiteado sin inventar contenido."""


class UnavailableLLMProvider:
    """Stub explicito para proveedores LLM aun no configurados."""

    name = "unavailable_llm"

    def decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        raise RuntimeError("No hay provider LLM configurado para cierre")

    def correct(self, raw_text: str, context_before: str = "", context_after: str = "") -> CorrectionResult:
        raise RuntimeError("No hay provider LLM configurado para correccion")
