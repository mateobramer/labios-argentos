"""Helpers visuales livianos para notebooks de segmentacion_oraciones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import HTML

from segmentacion_oraciones.src.cierre_ml import load_training_examples
from segmentacion_oraciones.src.entrenar_cierre import run_training_pipeline


PALETTE = {
    "ink": "#17202a",
    "muted": "#596573",
    "line": "#d8dee9",
    "green": "#138a63",
    "red": "#b42318",
    "blue": "#2563eb",
    "amber": "#b7791f",
    "bg": "#f7f8fb",
}


def find_default_synthetic_zip() -> Path | None:
    path = Path.home() / "Downloads" / "synthetic_sentence_boundary_es_ar_v3_curated.zip"
    return path if path.exists() else None


def default_training_inputs() -> list[str]:
    inputs = ["segmentacion_oraciones/ground_truth"]
    synthetic_zip = find_default_synthetic_zip()
    if synthetic_zip:
        inputs.append(str(synthetic_zip))
    return inputs


def ensure_training_summary(inputs: list[str], output_dir: str | Path, *, seed: int = 13) -> dict[str, Any]:
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_training_pipeline(inputs, output_dir=output_dir, seed=seed)


def examples_overview(inputs: list[str]) -> dict[str, Any]:
    examples = load_training_examples(inputs)
    actions: dict[str, int] = {}
    synthetic: dict[str, int] = {}
    splits: dict[str, int] = {}
    for example in examples:
        actions[example.expected_action] = actions.get(example.expected_action, 0) + 1
        synthetic[str(example.synthetic)] = synthetic.get(str(example.synthetic), 0) + 1
        split = str(example.metadata.get("input_split") or "real_ground_truth")
        splits[split] = splits.get(split, 0) + 1
    return {
        "examples": len(examples),
        "sources": len({example.source_id for example in examples}),
        "actions": dict(sorted(actions.items())),
        "synthetic": dict(sorted(synthetic.items())),
        "splits": dict(sorted(splits.items())),
    }


def candidate_frame(summary: dict[str, Any], split: str = "val") -> pd.DataFrame:
    rows = []
    for row in summary["candidates"]:
        metrics = row[split]
        rows.append(
            {
                "modelo": row["name"],
                "selection_score": metrics["selection_score"],
                "commit_f1": metrics["commit_f1"],
                "commit_precision": metrics["commit_precision"],
                "commit_recall": metrics["commit_recall"],
                "premature_commit_rate": metrics["premature_commit_rate"],
                "low_confidence_recall": metrics["low_confidence_recall"],
                "boundary_p95_abs": metrics["boundary_error_clips"]["p95_abs"],
                "latency_p50_ms": metrics["latency_ms"]["p50"],
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def card_grid(cards: list[tuple[str, str, str | None]]) -> HTML:
    html = [
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:12px 0 18px 0;'>"
    ]
    for title, value, subtitle in cards:
        html.append(
            "<div style='background:#fff;border:1px solid {line};border-radius:8px;padding:14px 16px;'>"
            "<div style='font-size:12px;color:{muted};text-transform:uppercase;letter-spacing:.04em'>{title}</div>"
            "<div style='font-size:28px;font-weight:700;color:{ink};line-height:1.2'>{value}</div>"
            "<div style='font-size:13px;color:{muted};min-height:18px'>{subtitle}</div>"
            "</div>".format(
                line=PALETTE["line"],
                muted=PALETTE["muted"],
                ink=PALETTE["ink"],
                title=title,
                value=value,
                subtitle=subtitle or "",
            )
        )
    html.append("</div>")
    return HTML("".join(html))


def note_box(text: str, *, kind: str = "info") -> HTML:
    colors = {
        "info": ("#eef5ff", "#2563eb"),
        "ok": ("#ecfdf5", "#138a63"),
        "warn": ("#fff7ed", "#b7791f"),
        "bad": ("#fff1f2", "#b42318"),
    }
    bg, border = colors.get(kind, colors["info"])
    return HTML(
        f"<div style='background:{bg};border-left:5px solid {border};padding:12px 14px;"
        f"border-radius:6px;margin:12px 0;color:{PALETTE['ink']};font-size:14px'>{text}</div>"
    )
