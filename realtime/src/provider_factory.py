"""Factory de providers sin dependencias obligatorias."""

from __future__ import annotations

from realtime.src.cierre import HeuristicClosureProvider
from realtime.src.corrector import IdentityCorrectionProvider
from realtime.src.llm.providers import ClosureProvider, CorrectionProvider


def make_closure_provider(
    name: str = "heuristic",
    *,
    model_path: str | None = None,
    ollama_model: str = "qwen3:4b",
    ollama_url: str = "http://localhost:11434",
    timeout_s: float = 2.5,
) -> ClosureProvider:
    if name == "heuristic":
        return HeuristicClosureProvider()
    if name == "linear":
        from realtime.src.cierre_ml import LinearClosureProvider

        if not model_path:
            raise ValueError("model_path es obligatorio para provider linear")
        return LinearClosureProvider.load(model_path)
    if name == "ollama":
        from realtime.src.llm.ollama_provider import OllamaProvider

        return OllamaProvider(model=ollama_model, base_url=ollama_url, timeout_s=timeout_s)
    raise ValueError(f"Provider de cierre desconocido: {name}")


def make_correction_provider(
    name: str = "identity",
    *,
    ollama_model: str = "qwen3:4b",
    ollama_url: str = "http://localhost:11434",
    timeout_s: float = 2.5,
) -> CorrectionProvider:
    if name == "identity":
        return IdentityCorrectionProvider()
    if name == "ollama":
        from realtime.src.llm.ollama_provider import OllamaProvider

        return OllamaProvider(model=ollama_model, base_url=ollama_url, timeout_s=timeout_s)
    raise ValueError(f"Provider de correccion desconocido: {name}")
