"""Export de datasets para entrenar un LLM/clasificador de cierre.

El modelo online no deberia generar texto libre: se lo entrena para devolver JSON
cerrado con action, confidence y una razon corta. Este modulo solo prepara datos;
el entrenamiento pesado queda para la VM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from segmentacion_oraciones.src.contracts import CommitAction
from segmentacion_oraciones.src.cierre_ml import ClosureTrainingExample, load_training_examples


SYSTEM_PROMPT = (
    "Sos un clasificador causal de cierre de oracion para VSR/RSV en espanol "
    "rioplatense. Recibis un buffer textual ruidoso y devolves solo JSON valido "
    "con action wait, commit o low_confidence. No corrijas texto ni inventes palabras."
)


def export_sft_jsonl(
    inputs: list[str | Path],
    output_path: str | Path,
    *,
    max_examples: int | None = None,
    seed: int = 13,
) -> Path:
    examples = load_training_examples(inputs)
    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        examples = rng.sample(examples, max_examples)
        examples.sort(key=lambda row: (row.source_id, row.order, row.clip_id))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for example in examples:
            fh.write(json.dumps(example_to_chat_row(example), ensure_ascii=False, sort_keys=True) + "\n")
    return output


def example_to_chat_row(example: ClosureTrainingExample) -> dict[str, Any]:
    return {
        "source_id": example.source_id,
        "clip_id": example.clip_id,
        "synthetic": example.synthetic,
        "metadata": {
            "input_split": example.metadata.get("input_split", ""),
            "noise_level": example.metadata.get("noise_level", ""),
            "difficulty": example.metadata.get("difficulty", ""),
            "context": example.metadata.get("context", ""),
            "buffer_clip_count": example.metadata.get("buffer_clip_count", 1),
            "current_noise_tags": example.metadata.get("current_noise_tags", []),
            "buffer_noise_tags": example.metadata.get("buffer_noise_tags", []),
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_input_payload(example), ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(_target_payload(example), ensure_ascii=False, sort_keys=True)},
        ],
    }


def _input_payload(example: ClosureTrainingExample) -> dict[str, Any]:
    return {
        "visible_context": example.partial_text,
        "buffer_clip_count": example.metadata.get("buffer_clip_count", 1),
        "noise_tags": example.metadata.get("buffer_noise_tags", []),
        "noise_level": example.metadata.get("noise_level", ""),
    }


def _target_payload(example: ClosureTrainingExample) -> dict[str, Any]:
    return {
        "action": example.expected_action,
        "committed_text": example.committed_text if example.expected_action == CommitAction.COMMIT.value else "",
        "committed_sentence_id": example.sentence_id or None,
        "confidence": 1.0 if example.expected_action != CommitAction.LOW_CONFIDENCE.value else 0.55,
        "reason": "label_supervisado_de_cierre",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta JSONL chat/SFT para entrenar cierre con un LLM chico.")
    parser.add_argument("--input", action="append", dest="inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    output = export_sft_jsonl(args.inputs, args.output, max_examples=args.max_examples, seed=args.seed)
    print(json.dumps({"output": str(output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
