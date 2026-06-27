"""Simulador offline del flujo realtime sin GPU ni servicios externos."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

from realtime.src.contracts import CommitAction, FeedbackEvent, PartialHypothesis
from realtime.src.feedback import FeedbackWriter
from realtime.src.logging_utils import JSONLLogger, provider_log_event
from realtime.src.provider_factory import make_closure_provider, make_correction_provider
from realtime.src.validation import validate_commit_decision, validate_correction_result


DEMO_TEXTS = [
    "",
    "yo creo que",
    "hola como estas todo bien gracias por venir hoy",
    "bueno bueno bueno bueno",
    "me parece que",
    "ayer fuimos a la cancha y estuvo buenisimo.",
    "no se si",
    "vos tenes razon che gracias por avisarme",
]


def run_simulation(
    texts: Iterable[str],
    *,
    closure_provider_name: str = "heuristic",
    correction_provider_name: str = "identity",
    model_path: str | None = None,
    ollama_model: str = "qwen3:4b",
    ollama_url: str = "http://localhost:11434",
    timeout_s: float = 2.5,
    log_path: str | Path = "realtime/outputs/llm_logs/events.jsonl",
    feedback_path: str | Path = "realtime/outputs/feedback/events.jsonl",
) -> dict[str, object]:
    closure = make_closure_provider(
        closure_provider_name,
        model_path=model_path,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        timeout_s=timeout_s,
    )
    corrector = make_correction_provider(
        correction_provider_name,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        timeout_s=timeout_s,
    )
    logger = JSONLLogger(log_path)
    feedback = FeedbackWriter(feedback_path)

    counts = {CommitAction.COMMIT.value: 0, CommitAction.WAIT.value: 0, CommitAction.LOW_CONFIDENCE.value: 0}
    latencies: dict[str, list[float]] = {
        "closure": [],
        "correction": [],
        "validation": [],
        "logging": [],
    }
    fallback_count = 0
    examples = list(texts)

    for idx, text in enumerate(examples, start=1):
        segment_id = f"sim_{idx:04d}"
        hypothesis = PartialHypothesis(partial_text=text, segment_id=segment_id, source="simulation")

        started = time.perf_counter()
        decision = closure.decide(hypothesis)
        closure_ms = _elapsed_ms(started)
        latencies["closure"].append(closure_ms)

        started = time.perf_counter()
        decision, fallback_used = validate_commit_decision(decision)
        validation_ms = _elapsed_ms(started)
        latencies["validation"].append(validation_ms)
        provider_fallback = "fallback" in decision.risk_flags
        fallback_count += int(fallback_used or provider_fallback)
        counts[decision.action.value] += 1

        correction_ms = 0.0
        correction = None
        correction_fallback = False
        if decision.action == CommitAction.COMMIT:
            started = time.perf_counter()
            correction = corrector.correct(decision.committed_text)
            correction_ms = _elapsed_ms(started)
            latencies["correction"].append(correction_ms)

            started = time.perf_counter()
            correction, correction_fallback = validate_correction_result(correction, decision.committed_text)
            latencies["validation"].append(_elapsed_ms(started))
            correction_provider_fallback = "fallback" in correction.risk_flags
            fallback_count += int(correction_fallback or correction_provider_fallback)

            feedback.write(
                FeedbackEvent(
                    segment_id=segment_id,
                    raw_vsr_text=text,
                    committed_text=decision.committed_text,
                    corrected_text=correction.corrected_text,
                    user_decision="unclear",
                    latency_ms=closure_ms + correction_ms,
                    source="simulation",
                    review_status="pending",
                )
            )

        log_event = provider_log_event(
            provider=closure.name,
            stage="closure",
            input_payload=hypothesis,
            output_payload=decision,
            latency_ms=closure_ms,
            valid=not fallback_used,
            fallback_used=fallback_used or provider_fallback,
        )
        ok, logging_ms, error = logger.write(log_event)
        latencies["logging"].append(logging_ms)
        if not ok:
            fallback_count += 1

        if correction is not None:
            log_event = provider_log_event(
                provider=corrector.name,
                stage="correction",
                input_payload={"raw_text": decision.committed_text},
                output_payload=correction,
                latency_ms=correction_ms,
                valid=not correction_fallback,
                fallback_used=correction_fallback or ("fallback" in correction.risk_flags),
                error=error,
            )
            ok, logging_ms, _ = logger.write(log_event)
            latencies["logging"].append(logging_ms)
            if not ok:
                fallback_count += 1

    return {
        "count": len(examples),
        "providers": {
            "closure": getattr(closure, "name", closure.__class__.__name__),
            "correction": getattr(corrector, "name", corrector.__class__.__name__),
        },
        "actions": counts,
        "latency_ms": {name: _latency_summary(values) for name, values in latencies.items()},
        "fallbacks": fallback_count,
    }


def load_texts(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 4),
        "p95": round(_percentile(ordered, 95), 4),
    }


def _percentile(ordered_values: list[float], percentile: int) -> float:
    if not ordered_values:
        return 0.0
    index = max(0, min(len(ordered_values) - 1, round((percentile / 100) * (len(ordered_values) - 1))))
    return ordered_values[index]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula el flujo realtime sin GPU ni LLM externo.")
    parser.add_argument("--demo", action="store_true", help="Usar frases sinteticas incluidas.")
    parser.add_argument("--input", help="Archivo de texto, una hipotesis parcial por linea.")
    parser.add_argument("--closure-provider", choices=["heuristic", "linear", "ollama"], default="heuristic")
    parser.add_argument("--correction-provider", choices=["identity", "ollama"], default="identity")
    parser.add_argument("--model-path", help="Modelo JSON para --closure-provider linear.")
    parser.add_argument("--ollama-model", default="qwen3:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout-s", type=float, default=2.5)
    parser.add_argument("--log-path", default="realtime/outputs/llm_logs/events.jsonl")
    parser.add_argument("--feedback-path", default="realtime/outputs/feedback/events.jsonl")
    args = parser.parse_args()

    if args.input:
        texts = load_texts(args.input)
    elif args.demo:
        texts = DEMO_TEXTS
    else:
        parser.error("Usar --demo o --input")

    summary = run_simulation(
        texts,
        closure_provider_name=args.closure_provider,
        correction_provider_name=args.correction_provider,
        model_path=args.model_path,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        timeout_s=args.timeout_s,
        log_path=args.log_path,
        feedback_path=args.feedback_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
