"""Helpers de lectura para el notebook batch VSR."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"


def cargar_configs(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    rows = []
    for path in sorted((output_base / "experiments").glob("*/experiment_config.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def tamanos_experimentos(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, config in configs.iterrows():
        train = pd.read_csv(config["train_split"]) if Path(config["train_split"]).exists() else pd.DataFrame()
        val = pd.read_csv(config["val_split"]) if Path(config["val_split"]).exists() else pd.DataFrame()
        test = pd.read_csv(config["test_split"]) if Path(config["test_split"]).exists() else pd.DataFrame()
        rows.append(
            {
                "experiment": config["experiment"],
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
        )
    return pd.DataFrame(rows)


def transcript_changes(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "transcript_cleaning_changes.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def transcript_summary(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return pd.DataFrame([{"transcripts": 0, "changed": 0, "unchanged": 0}])
    changed = int((changes["changed"].astype(str).str.lower() == "true").sum())
    return pd.DataFrame(
        [
            {
                "transcripts": len(changes),
                "changed": changed,
                "unchanged": len(changes) - changed,
            }
        ]
    )


def preprocessing_smoke(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "preprocessing_variant_manifest_smoke.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def resultados(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "results" / "summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
