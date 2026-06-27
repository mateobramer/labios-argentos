"""Genera planes de variaciones para datos sinteticos de cierre."""

from __future__ import annotations

import argparse
import csv
import json
import random
from itertools import product
from pathlib import Path
from typing import Any


CONTEXTS = [
    "streaming",
    "universidad",
    "trabajo",
    "futbol",
    "pareja",
    "politica informal",
    "familia",
    "entrevista",
    "clase",
    "salud",
]
REGISTERS = ["muy informal", "informal", "neutro", "tecnico simple"]
SPEAKERS = ["joven", "adulto", "profesor", "streamer", "estudiante", "profesional"]
NOISE_LEVELS = ["bajo", "medio", "alto"]
DIFFICULTIES = [
    "oraciones largas",
    "muletillas",
    "interrupciones",
    "repeticiones",
    "conectores colgantes",
    "cambio brusco de tema",
    "frases incompletas",
]


def build_variations(*, seed: int = 13, max_items: int | None = None, shuffle: bool = False) -> list[dict[str, Any]]:
    rows = [
        {
            "variation_id": f"var_{idx:04d}",
            "context": context,
            "register": register,
            "speaker": speaker,
            "noise_level": noise,
            "difficulty": difficulty,
            "recommended_clips": 180 if noise == "alto" else 240,
        }
        for idx, (context, register, speaker, noise, difficulty) in enumerate(
            product(CONTEXTS, REGISTERS, SPEAKERS, NOISE_LEVELS, DIFFICULTIES),
            start=1,
        )
    ]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if max_items is not None:
        rows = rows[:max_items]
        for idx, row in enumerate(rows, start=1):
            row["variation_id"] = f"var_{idx:04d}"
    return rows


def write_plan(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".json":
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif suffix == ".csv":
        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    else:
        with output.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera combinaciones para pedir data sintetica a GPT Pro.")
    parser.add_argument("--output", default="realtime/outputs/synthetic_plan/variations.jsonl")
    parser.add_argument("--max", type=int, help="Limita la cantidad de variaciones.")
    parser.add_argument("--shuffle", action="store_true", help="Baraja antes de limitar.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = build_variations(seed=args.seed, max_items=args.max, shuffle=args.shuffle)
    output = write_plan(rows, args.output)
    print(json.dumps({"output": str(output), "variations": len(rows)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
