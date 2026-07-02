"""Exporta la muestra visual a un scenario compatible con el evaluador de Gimeno."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


RAIZ_REPO = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = RAIZ_REPO / "data" / "metadata" / "visual_quality_vsr_eval_sample.csv"
DEFAULT_OUTPUT_BASE = RAIZ_REPO / "evaluation" / "data" / "visual_quality_vsr_eval"
DEFAULT_MAPPING = RAIZ_REPO / "data" / "metadata" / "visual_quality_vsr_eval_mapping.csv"
DEFAULT_DB = "Rioplatense"
DEFAULT_SCENARIO = "visual-quality-sample"
SCENARIO_SPLIT = "test"

REQUIRED_COLUMNS = {
    "source_id",
    "clip",
    "split",
    "path_roi",
    "path_text",
    "training_usability",
}

MAPPING_COLUMNS = [
    "sample_id",
    "source_id",
    "clip",
    "split",
    "scenario_split",
    "path_roi",
    "path_text",
    "policy_moderate",
    "training_usability",
    "review_score",
    "quality_score",
]


def leer_sample(path: str | Path) -> list[dict[str, str]]:
    path = repo_path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        validar_columnas(reader.fieldnames or [])
        return list(reader)


def validar_columnas(fieldnames: list[str]) -> None:
    disponibles = set(fieldnames)
    faltantes = sorted(REQUIRED_COLUMNS - disponibles)
    if "policy_moderate_v2" not in disponibles and "policy_moderate" not in disponibles:
        faltantes.append("policy_moderate_v2")
    if faltantes:
        raise ValueError(f"CSV de muestra sin columnas obligatorias: {', '.join(faltantes)}")


def exportar_visual_quality_scenario(
    sample_csv: str | Path = DEFAULT_SAMPLE,
    *,
    output_base: str | Path = DEFAULT_OUTPUT_BASE,
    mapping_output: str | Path = DEFAULT_MAPPING,
    database: str = DEFAULT_DB,
    scenario: str = DEFAULT_SCENARIO,
) -> dict[str, Any]:
    rows = leer_sample(sample_csv)
    validar_paths(rows)

    output_base = repo_path(output_base)
    base_db = output_base / database
    rois_dir = base_db / "ROIs"
    text_dir = base_db / "transcriptions"
    split_dir = base_db / "splits" / scenario
    split_csv = split_dir / f"{SCENARIO_SPLIT}{database}.csv"
    for path in (rois_dir, text_dir, split_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows_ordenadas = sorted(rows, key=lambda row: (row["source_id"], row["clip"], row["split"]))
    speakers = asignar_speakers(rows_ordenadas)
    indices_por_speaker: dict[str, int] = defaultdict(int)
    split_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []

    for row in rows_ordenadas:
        source_id = row["source_id"]
        spk = speakers[source_id]
        idx = indices_por_speaker[spk]
        indices_por_speaker[spk] += 1
        sample_id = f"{spk}_{idx:04d}"

        src_roi = repo_path(row["path_roi"])
        src_text = repo_path(row["path_text"])
        dst_roi = rois_dir / spk / f"{sample_id}.npz"
        dst_text = text_dir / spk / f"{sample_id}.txt"
        dst_roi.parent.mkdir(parents=True, exist_ok=True)
        dst_text.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_roi, dst_roi)
        texto = src_text.read_text(encoding="utf-8").strip()
        dst_text.write_text(texto + "\n", encoding="utf-8")

        split_rows.append({"sampleID": sample_id})
        mapping_rows.append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "clip": row["clip"],
                "split": row["split"],
                "scenario_split": SCENARIO_SPLIT,
                "path_roi": repo_rel(src_roi),
                "path_text": repo_rel(src_text),
                "policy_moderate": row.get("policy_moderate_v2") or row.get("policy_moderate") or "",
                "training_usability": row["training_usability"],
                "review_score": row.get("review_score", ""),
                "quality_score": row.get("quality_score", ""),
            }
        )

    escribir_csv(split_rows, split_csv, ["sampleID"])
    mapping_path = escribir_csv(mapping_rows, mapping_output, MAPPING_COLUMNS)
    return {
        "database": database,
        "scenario": scenario,
        "clips": len(mapping_rows),
        "fuentes": len(speakers),
        "scenario_dir": str(base_db),
        "split_csv": str(split_csv),
        "mapping_csv": str(mapping_path),
    }


def asignar_speakers(rows: list[dict[str, str]]) -> dict[str, str]:
    sources = sorted({row["source_id"] for row in rows})
    return {source_id: f"vq{idx:02d}" for idx, source_id in enumerate(sources, start=1)}


def validar_paths(rows: list[dict[str, str]]) -> None:
    faltantes = []
    for row in rows:
        for col in ("path_roi", "path_text"):
            path = repo_path(row[col])
            if not path.exists():
                faltantes.append(f"{col}={row[col]}")
    if faltantes:
        ejemplo = "; ".join(faltantes[:8])
        extra = "" if len(faltantes) <= 8 else f"; ... ({len(faltantes)} faltantes)"
        raise FileNotFoundError(f"Paths faltantes para exportar scenario: {ejemplo}{extra}")


def escribir_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str]) -> Path:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return RAIZ_REPO / path


def repo_rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(RAIZ_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar muestra visual a scenario VSR de Gimeno.")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="CSV visual_quality_vsr_eval_sample.csv")
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE), help="Directorio que contiene <DB>/")
    parser.add_argument("--mapping-output", default=str(DEFAULT_MAPPING), help="CSV de trazabilidad sample_id -> clip")
    parser.add_argument("--database", default=DEFAULT_DB)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    args = parser.parse_args()

    summary = exportar_visual_quality_scenario(
        args.sample,
        output_base=args.output_base,
        mapping_output=args.mapping_output,
        database=args.database,
        scenario=args.scenario,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
