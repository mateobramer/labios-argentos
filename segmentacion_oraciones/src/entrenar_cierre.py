"""CLI para entrenar y comparar modelos livianos de cierre."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from segmentacion_oraciones.src.cierre import HeuristicClosureProvider
from segmentacion_oraciones.src.cierre_ml import (
    evaluate_provider,
    load_training_examples,
    make_majority_provider,
    save_cases_jsonl,
    split_examples,
    train_linear_model,
)


DEFAULT_INPUTS = ["segmentacion_oraciones/ground_truth"]
DEFAULT_AMOR_TEST = "segmentacion_oraciones/ground_truth/charla_amor_desamor.json"


MODEL_CONFIGS = [
    {
        "name": "linear_text",
        "include_heuristic": False,
        "class_balance": False,
        "epochs": 8,
    },
    {
        "name": "linear_text_balanced",
        "include_heuristic": False,
        "class_balance": True,
        "epochs": 12,
    },
    {
        "name": "linear_heuristic",
        "include_heuristic": True,
        "class_balance": False,
        "epochs": 8,
    },
    {
        "name": "linear_heuristic_balanced",
        "include_heuristic": True,
        "class_balance": True,
        "epochs": 12,
    },
]


def run_training_pipeline(
    inputs: list[str],
    *,
    output_dir: str | Path,
    seed: int = 13,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    examples = load_training_examples(inputs)
    splits = split_examples(examples, seed=seed)
    for split_name, split_examples_list in splits.items():
        save_cases_jsonl(split_examples_list, output / f"{split_name}_cases.jsonl")

    candidates: list[dict[str, Any]] = []

    baselines = [
        ("majority", make_majority_provider(splits["train"])),
        ("heuristic", HeuristicClosureProvider()),
    ]
    for name, provider in baselines:
        candidates.append(_evaluate_candidate(name, provider, splits, output))

    for config in MODEL_CONFIGS:
        provider = train_linear_model(
            splits["train"],
            include_heuristic=bool(config["include_heuristic"]),
            class_balance=bool(config["class_balance"]),
            epochs=int(config["epochs"]),
            seed=seed,
            name=str(config["name"]),
        )
        model_path = output / f"{config['name']}.model.json"
        provider.save(model_path)
        candidate = _evaluate_candidate(str(config["name"]), provider, splits, output)
        candidate["model_path"] = str(model_path)
        candidates.append(candidate)

    best = max(candidates, key=lambda row: (row["val"]["selection_score"], row["val"]["commit_f1"], row["val"]["accuracy"]))
    best_config_path = output / "best_config.json"
    best_payload = {
        "best_name": best["name"],
        "best_model_path": best.get("model_path"),
        "selection_metric": "val.selection_score",
        "selection_score": best["val"]["selection_score"],
        "commit_f1": best["val"]["commit_f1"],
        "premature_commit_rate": best["val"]["premature_commit_rate"],
        "low_confidence_recall": best["val"]["low_confidence_recall"],
        "boundary_error_clips": best["val"]["boundary_error_clips"],
        "overcommit_risk_rate": best["val"]["overcommit_risk_rate"],
        "note": "Si best_model_path es null, gano un baseline no entrenable.",
    }
    best_config_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "inputs": inputs,
        "output_dir": str(output),
        "examples": len(examples),
        "sources": sorted({example.source_id for example in examples}),
        "dataset_summary": _dataset_summary(examples),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "best": best_payload,
        "candidates": candidates,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_synthetic_to_real_pipeline(
    train_inputs: list[str],
    *,
    test_input: str | Path = DEFAULT_AMOR_TEST,
    output_dir: str | Path,
    seed: int = 13,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_examples = [example for example in load_training_examples(train_inputs) if example.synthetic]
    if not train_examples:
        raise ValueError("No se cargaron ejemplos sinteticos para entrenar")
    test_examples = load_training_examples([test_input])

    internal_splits = split_examples(train_examples, seed=seed)
    splits = {
        "train": internal_splits["train"],
        "val": internal_splits["val"],
        "test": test_examples,
    }
    for split_name, split_examples_list in splits.items():
        save_cases_jsonl(split_examples_list, output / f"{split_name}_cases.jsonl")

    candidates: list[dict[str, Any]] = []
    baselines = [
        ("majority", make_majority_provider(splits["train"])),
        ("heuristic", HeuristicClosureProvider()),
    ]
    for name, provider in baselines:
        candidates.append(_evaluate_candidate(name, provider, splits, output))

    for config in MODEL_CONFIGS:
        provider = train_linear_model(
            splits["train"],
            include_heuristic=bool(config["include_heuristic"]),
            class_balance=bool(config["class_balance"]),
            epochs=int(config["epochs"]),
            seed=seed,
            name=str(config["name"]),
        )
        model_path = output / f"{config['name']}.model.json"
        provider.save(model_path)
        candidate = _evaluate_candidate(str(config["name"]), provider, splits, output)
        candidate["model_path"] = str(model_path)
        candidates.append(candidate)

    best = max(candidates, key=lambda row: (row["test"]["selection_score"], row["test"]["commit_f1"], row["test"]["accuracy"]))
    best_payload = {
        "best_name": best["name"],
        "best_model_path": best.get("model_path"),
        "selection_metric": "test.selection_score",
        "selection_score": best["test"]["selection_score"],
        "commit_f1": best["test"]["commit_f1"],
        "premature_commit_rate": best["test"]["premature_commit_rate"],
        "low_confidence_recall": best["test"]["low_confidence_recall"],
        "boundary_error_clips": best["test"]["boundary_error_clips"],
        "overcommit_risk_rate": best["test"]["overcommit_risk_rate"],
        "note": "Entrenado solo con sintetico; test externo en charla_amor_desamor.",
    }
    (output / "best_config.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "mode": "synthetic_train_real_amor_test",
        "train_inputs": train_inputs,
        "test_input": str(test_input),
        "output_dir": str(output),
        "examples": len(train_examples) + len(test_examples),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "sources": sorted({example.source_id for example in train_examples + test_examples}),
        "dataset_summary": {
            "train": _dataset_summary(train_examples),
            "test": _dataset_summary(test_examples),
        },
        "splits": {name: len(rows) for name, rows in splits.items()},
        "best": best_payload,
        "candidates": candidates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _evaluate_candidate(name: str, provider, splits: dict[str, list], output: Path) -> dict[str, Any]:
    result = {"name": name}
    for split_name, examples in splits.items():
        metrics = evaluate_provider(provider, examples)
        compact = _compact_metrics(metrics)
        result[split_name] = compact
        metrics_path = output / f"{name}.{split_name}.metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "count",
        "provider",
        "accuracy",
        "macro_f1",
        "commit_precision",
        "commit_recall",
        "commit_f1",
        "premature_commit_rate",
        "unnecessary_wait_rate",
        "low_confidence_recall",
        "boundary_error_clips",
        "early_commit_rate_by_boundary",
        "late_commit_rate_by_boundary",
        "on_time_commit_rate_by_boundary",
        "overcommit_risk_rate",
        "selection_score",
        "latency_ms",
        "fallbacks",
    ]
    return {key: metrics[key] for key in keep}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y compara modelos livianos de cierre causal.")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="JSON, ZIP o carpeta con ground truth real/sintetico. Se puede repetir.",
    )
    parser.add_argument(
        "--synthetic-train-real-test",
        action="store_true",
        help="Entrena solo con sintetico y evalua test externo sobre charla_amor_desamor.",
    )
    parser.add_argument(
        "--test-input",
        default=DEFAULT_AMOR_TEST,
        help="Ground truth real usado como test externo.",
    )
    parser.add_argument("--output-dir", default="segmentacion_oraciones/outputs/cierre_training")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    inputs = args.inputs or DEFAULT_INPUTS
    if args.synthetic_train_real_test:
        summary = run_synthetic_to_real_pipeline(
            inputs,
            test_input=args.test_input,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        summary = run_training_pipeline(inputs, output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(_printable_summary(summary), ensure_ascii=False, indent=2, sort_keys=True))


def _printable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": summary["output_dir"],
        "examples": summary["examples"],
        "splits": summary["splits"],
        "dataset_summary": summary["dataset_summary"],
        "best": summary["best"],
        "candidates": [
            {
                "name": row["name"],
                "val_selection_score": row["val"]["selection_score"],
                "val_commit_f1": row["val"]["commit_f1"],
                "val_premature_commit_rate": row["val"]["premature_commit_rate"],
                "val_low_confidence_recall": row["val"]["low_confidence_recall"],
                "val_boundary_error_clips": row["val"]["boundary_error_clips"],
                "test_selection_score": row["test"]["selection_score"],
            }
            for row in summary["candidates"]
        ],
    }


def _dataset_summary(examples: list) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for key in ("synthetic", "input_split", "noise_level", "difficulty", "context"):
        counts: dict[str, int] = {}
        for example in examples:
            value = str(example.metadata.get(key, example.synthetic if key == "synthetic" else ""))
            if value:
                counts[value] = counts.get(value, 0) + 1
        if counts:
            summary[key] = dict(sorted(counts.items()))
    return summary


if __name__ == "__main__":
    main()
