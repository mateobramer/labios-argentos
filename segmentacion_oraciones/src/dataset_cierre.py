"""Casos livianos de evaluacion de cierre desde textos del proyecto."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from segmentacion_oraciones.src.cierre import HeuristicClosureProvider


def build_cases_from_text(
    text: str,
    *,
    segment_id: str,
    source: str = "",
    max_cases: int = 3,
) -> list[dict[str, object]]:
    """Genera casos conservadores desde una referencia textual.

    No pretende crear ground truth perfecto de streaming. Sirve para armar un
    benchmark liviano y reproducible: prefijos incompletos deberian esperar, y
    frases suficientemente largas completas pueden commitearse.
    """

    tokens = [token for token in text.split() if token]
    if not tokens:
        return []

    cases: list[dict[str, object]] = []

    if len(tokens) >= 3:
        prefix_len = min(4, len(tokens) - 1)
        cases.append(
            _case(
                partial_text=" ".join(tokens[:prefix_len]),
                expected_action="wait",
                segment_id=f"{segment_id}_prefix",
                source=source,
                case="prefix_wait",
            )
        )

    dangling_case = _dangling_prefix_case(tokens, segment_id, source)
    if dangling_case is not None:
        cases.append(dangling_case)

    if len(tokens) >= 8:
        cases.append(
            _case(
                partial_text=" ".join(tokens),
                expected_action="commit",
                segment_id=f"{segment_id}_full",
                source=source,
                case="full_commit",
            )
        )

    return _dedupe_cases(cases)[:max_cases]


def build_cases_from_split_csv(path: str | Path, *, limit: int = 30) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            text = row.get("texto", "")
            clip = row.get("clip", f"row_{idx:04d}")
            source = f"{split_path.as_posix()}:{row.get('split', '')}:{row.get('titulo', '')}:{clip}"
            segment_id = f"{row.get('split', 'split')}_{idx:04d}_{clip}"
            cases.extend(build_cases_from_text(text, segment_id=segment_id, source=source))
            if len(cases) >= limit:
                return cases[:limit]
    return cases[:limit]


def write_jsonl(cases: Iterable[dict[str, object]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def _dangling_prefix_case(tokens: list[str], segment_id: str, source: str) -> dict[str, object] | None:
    dangling = HeuristicClosureProvider.dangling_tokens
    # Buscar un prefijo con conector al final y al menos una palabra despues
    # en la referencia completa, asi la etiqueta wait es razonable.
    for idx in range(2, min(len(tokens), 10)):
        if tokens[idx - 1].lower() in dangling and idx < len(tokens):
            return _case(
                partial_text=" ".join(tokens[:idx]),
                expected_action="wait",
                segment_id=f"{segment_id}_dangling_{idx}",
                source=source,
                case="dangling_wait",
            )
    return None


def _case(
    *,
    partial_text: str,
    expected_action: str,
    segment_id: str,
    source: str,
    case: str,
) -> dict[str, object]:
    return {
        "partial_text": partial_text,
        "expected_action": expected_action,
        "segment_id": segment_id,
        "source": source,
        "case": case,
    }


def _dedupe_cases(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    deduped = []
    for case in cases:
        key = (str(case["partial_text"]), str(case["expected_action"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera casos JSONL de cierre desde un split CSV.")
    parser.add_argument("--split", required=True, help="Ruta a vsr_models/splits/{train,val,test}.csv")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases = build_cases_from_split_csv(args.split, limit=args.limit)
    output = write_jsonl(cases, args.output)
    print(json.dumps({"output": str(output), "cases": len(cases)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
