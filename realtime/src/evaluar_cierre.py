"""Evaluacion offline del modulo de cierre."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from realtime.src.contracts import CommitAction, PartialHypothesis
from realtime.src.provider_factory import make_closure_provider
from realtime.src.validation import validate_commit_decision


DEMO_CASES = [
    {"partial_text": "", "expected_action": "low_confidence", "case": "empty"},
    {"partial_text": "yo creo que", "expected_action": "wait", "case": "dangling"},
    {"partial_text": "me parece que", "expected_action": "wait", "case": "dangling"},
    {"partial_text": "bueno bueno bueno bueno", "expected_action": "low_confidence", "case": "repetition"},
    {
        "partial_text": "ayer fuimos a la cancha y estuvo buenisimo.",
        "expected_action": "commit",
        "case": "punctuated",
    },
    {
        "partial_text": "vos tenes razon che gracias por avisarme hoy",
        "expected_action": "commit",
        "case": "long_enough",
    },
    {"partial_text": "no se si", "expected_action": "wait", "case": "incomplete"},
]


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "partial_text" not in data or "expected_action" not in data:
                raise ValueError("Cada linea debe tener partial_text y expected_action")
            cases.append(data)
    return cases


def evaluate_closure(cases: list[dict[str, Any]], provider) -> dict[str, Any]:
    rows = []
    latencies = []
    fallbacks = 0

    for idx, case in enumerate(cases, start=1):
        hypothesis = PartialHypothesis(
            partial_text=str(case["partial_text"]),
            segment_id=str(case.get("segment_id") or f"eval_{idx:04d}"),
            source=str(case.get("source") or "closure_eval"),
            vsr_confidence=case.get("vsr_confidence"),
            ms_since_last_commit=case.get("ms_since_last_commit"),
        )
        started = time.perf_counter()
        decision = provider.decide(hypothesis)
        latency_ms = _elapsed_ms(started)
        latencies.append(latency_ms)

        decision, used_fallback = validate_commit_decision(decision)
        provider_fallback = "fallback" in decision.risk_flags
        fallbacks += int(used_fallback or provider_fallback)
        expected = CommitAction(str(case["expected_action"]))
        rows.append(
            {
                "segment_id": hypothesis.segment_id,
                "case": case.get("case", ""),
                "expected": expected.value,
                "predicted": decision.action.value,
                "correct": expected == decision.action,
                "latency_ms": latency_ms,
                "fallback": used_fallback or provider_fallback,
                "reason": decision.reason,
                "risk_flags": decision.risk_flags,
            }
        )

    return {
        "count": len(rows),
        "provider": getattr(provider, "name", provider.__class__.__name__),
        "accuracy": _safe_div(sum(1 for row in rows if row["correct"]), len(rows)),
        "commit_precision": _precision(rows, "commit"),
        "commit_recall": _recall(rows, "commit"),
        "premature_commit_rate": _safe_div(
            sum(1 for row in rows if row["predicted"] == "commit" and row["expected"] != "commit"),
            sum(1 for row in rows if row["expected"] != "commit"),
        ),
        "unnecessary_wait_rate": _safe_div(
            sum(1 for row in rows if row["predicted"] == "wait" and row["expected"] == "commit"),
            sum(1 for row in rows if row["expected"] == "commit"),
        ),
        "low_confidence_recall": _recall(rows, "low_confidence"),
        "latency_ms": _latency_summary(latencies),
        "fallbacks": fallbacks,
        "rows": rows,
    }


def _precision(rows: list[dict[str, Any]], action: str) -> float:
    predicted = [row for row in rows if row["predicted"] == action]
    if not predicted:
        return 0.0
    return _safe_div(sum(1 for row in predicted if row["expected"] == action), len(predicted))


def _recall(rows: list[dict[str, Any]], action: str) -> float:
    expected = [row for row in rows if row["expected"] == action]
    if not expected:
        return 0.0
    return _safe_div(sum(1 for row in expected if row["predicted"] == action), len(expected))


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 4),
        "p95": round(_percentile(ordered, 95), 4),
    }


def _percentile(ordered_values: list[float], percentile: int) -> float:
    index = max(0, min(len(ordered_values) - 1, round((percentile / 100) * (len(ordered_values) - 1))))
    return ordered_values[index]


def _safe_div(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua offline el cierre de oracion.")
    parser.add_argument("--demo", action="store_true", help="Usar casos sinteticos etiquetados.")
    parser.add_argument("--input", help="JSONL con partial_text y expected_action.")
    parser.add_argument("--provider", choices=["heuristic", "linear", "ollama"], default="heuristic")
    parser.add_argument("--model-path", help="Modelo JSON para --provider linear.")
    parser.add_argument("--ollama-model", default="qwen3:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout-s", type=float, default=2.5)
    parser.add_argument("--output", help="Ruta opcional para guardar resumen JSON.")
    args = parser.parse_args()

    if args.input:
        cases = load_cases(args.input)
    elif args.demo:
        cases = DEMO_CASES
    else:
        parser.error("Usar --demo o --input")

    provider = make_closure_provider(
        args.provider,
        model_path=args.model_path,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        timeout_s=args.timeout_s,
    )
    summary = evaluate_closure(cases, provider)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
