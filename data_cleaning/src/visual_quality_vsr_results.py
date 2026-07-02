"""Cruza inferencia VSR de Gimeno con el mapping de la muestra visual."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from data_cleaning.src.visual_quality_report import error_rate, partir_ref_hyp


RAIZ_REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = RAIZ_REPO / "data" / "metadata" / "visual_quality_vsr_eval_mapping.csv"
DEFAULT_OUTPUT = RAIZ_REPO / "data" / "metadata" / "visual_quality_vsr_eval_results.csv"

RESULT_COLUMNS = [
    "source_id",
    "clip",
    "split",
    "policy_moderate",
    "training_usability",
    "reference",
    "hypothesis",
    "wer",
    "cer",
]

REQUIRED_MAPPING_COLUMNS = {
    "sample_id",
    "source_id",
    "clip",
    "split",
    "policy_moderate",
    "training_usability",
}


def cargar_mapping(path: str | Path) -> list[dict[str, str]]:
    path = repo_path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        faltantes = sorted(REQUIRED_MAPPING_COLUMNS - set(reader.fieldnames or []))
        if faltantes:
            raise ValueError(f"Mapping sin columnas obligatorias: {', '.join(faltantes)}")
        return list(reader)


def cargar_inf(path: str | Path) -> list[tuple[str, str]]:
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de inferencia: {path}")
    pares = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pares.append(partir_ref_hyp(line))
    return pares


def cruzar_predicciones(
    inf_path: str | Path,
    mapping_path: str | Path = DEFAULT_MAPPING,
    *,
    output_path: str | Path | None = DEFAULT_OUTPUT,
) -> list[dict[str, Any]]:
    mapping = cargar_mapping(mapping_path)
    pares = cargar_inf(inf_path)
    if len(mapping) != len(pares):
        raise ValueError(f"Cantidad incompatible: mapping={len(mapping)} predicciones={len(pares)}")

    rows = []
    for meta, (reference, hypothesis) in zip(mapping, pares):
        rows.append(
            {
                "source_id": meta["source_id"],
                "clip": meta["clip"],
                "split": meta["split"],
                "policy_moderate": meta["policy_moderate"],
                "training_usability": meta["training_usability"],
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": round(error_rate(reference.split(), hypothesis.split()), 6),
                "cer": round(error_rate(list(reference), list(hypothesis)), 6),
            }
        )

    if output_path is not None:
        escribir_csv(rows, output_path, RESULT_COLUMNS)
    return rows


def resumen_resultados(rows: list[dict[str, Any]], *, min_clips_por_grupo: int = 30) -> dict[str, Any]:
    por_policy = agrupar_metricas(rows, "policy_moderate")
    por_usability = agrupar_metricas(rows, "training_usability")
    return {
        "clips": len(rows),
        "wer_promedio": round(sum(float(row["wer"]) for row in rows) / max(len(rows), 1), 6),
        "cer_promedio": round(sum(float(row["cer"]) for row in rows) / max(len(rows), 1), 6),
        "por_policy_moderate": por_policy,
        "por_training_usability": por_usability,
        "warnings": warnings_grupos_chicos(
            [("policy_moderate", por_policy), ("training_usability", por_usability)],
            min_clips=min_clips_por_grupo,
        ),
    }


def agrupar_metricas(rows: list[dict[str, Any]], col: str) -> list[dict[str, Any]]:
    grupos: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grupos.setdefault(str(row.get(col, "")), []).append(row)
    salida = []
    for grupo, items in sorted(grupos.items()):
        salida.append(
            {
                col: grupo,
                "clips": len(items),
                "wer_promedio": round(sum(float(row["wer"]) for row in items) / max(len(items), 1), 6),
                "cer_promedio": round(sum(float(row["cer"]) for row in items) / max(len(items), 1), 6),
            }
        )
    return salida


def warnings_grupos_chicos(metricas: list[tuple[str, list[dict[str, Any]]]], *, min_clips: int) -> list[str]:
    warnings = []
    for col, rows in metricas:
        for row in rows:
            if int(row["clips"]) < min_clips:
                warnings.append(f"No concluir para {col}={row[col]}: solo {row['clips']} clips (<{min_clips}).")
    return warnings


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Cruzar test.inf de Gimeno con mapping visual y calcular WER/CER.")
    parser.add_argument("--inf", required=True, help="Archivo inference/test.inf producido por vsr_main.py")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rows = cruzar_predicciones(args.inf, args.mapping, output_path=args.output)
    print(json.dumps({"output": str(repo_path(args.output)), "resumen": resumen_resultados(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
