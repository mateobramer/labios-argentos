"""Muestreo estratificado para evaluar impacto VSR de la auditoria visual."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RAIZ_REPO = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_ANALYSIS = RAIZ_REPO / "data" / "metadata" / "visual_quality_policy_analysis_v2.csv"
DEFAULT_OUTPUT = RAIZ_REPO / "data" / "metadata" / "visual_quality_vsr_eval_sample.csv"

COLUMNAS_MUESTRA = [
    "source_id",
    "clip",
    "split",
    "path_roi",
    "path_text",
    "policy_moderate_v2",
    "training_usability",
    "review_severity",
    "review_score",
    "exclusion_reasons",
    "quality_score",
    "mouth_activity_score",
    "mouth_visibility_score",
    "scene_cut_score",
    "blur_score",
]


def leer_csv(path: str | Path) -> list[dict[str, str]]:
    path = repo_path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or COLUMNAS_MUESTRA
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def generar_muestra_estratificada(
    rows: list[dict[str, Any]],
    *,
    per_group: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Muestrea por policy/usability y balancea internamente por split/fuente."""

    if per_group <= 0:
        raise ValueError("per_group debe ser mayor a cero")

    grupos: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        policy = valor_policy_moderate_v2(row)
        usability = str(row.get("training_usability") or "")
        if not policy or not usability:
            continue
        grupos[(policy, usability)].append(row)

    salida: list[dict[str, Any]] = []
    for idx, key in enumerate(sorted(grupos)):
        candidatos = sorted(grupos[key], key=clave_estable)
        if len(candidatos) <= per_group:
            elegidos = candidatos
        else:
            elegidos = muestrear_balanceado(candidatos, per_group, seed=seed + idx)
        salida.extend(normalizar_fila(row) for row in elegidos)

    return sorted(salida, key=lambda row: (row["policy_moderate_v2"], row["training_usability"], row["split"], row["source_id"], row["clip"]))


def muestrear_balanceado(rows: list[dict[str, Any]], n: int, *, seed: int) -> list[dict[str, Any]]:
    """Round-robin determinista por split/source_id."""

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("split") or ""), str(row.get("source_id") or row.get("titulo") or ""))].append(row)

    for bucket_rows in buckets.values():
        bucket_rows.sort(key=clave_estable)
        rng.shuffle(bucket_rows)

    keys = sorted(buckets)
    rng.shuffle(keys)
    elegidos: list[dict[str, Any]] = []
    while len(elegidos) < n and keys:
        siguiente_keys = []
        for key in keys:
            if len(elegidos) >= n:
                break
            bucket = buckets[key]
            if bucket:
                elegidos.append(bucket.pop())
            if bucket:
                siguiente_keys.append(key)
        keys = siguiente_keys
    return elegidos


def resumen_muestra(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clips": len(rows),
        "policy_moderate_v2": dict(sorted(Counter(row.get("policy_moderate_v2", "") for row in rows).items())),
        "training_usability": dict(sorted(Counter(row.get("training_usability", "") for row in rows).items())),
        "split": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "fuentes": len({row.get("source_id", "") for row in rows}),
        "grupos_policy_usability": dict(
            sorted(Counter(f"{row.get('policy_moderate_v2','')}|{row.get('training_usability','')}" for row in rows).items())
        ),
    }


def normalizar_fila(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id") or row.get("titulo") or "",
        "clip": row.get("clip", ""),
        "split": row.get("split", ""),
        "path_roi": row.get("path_roi") or row.get("npz") or "",
        "path_text": row.get("path_text") or "",
        "policy_moderate_v2": valor_policy_moderate_v2(row),
        "training_usability": row.get("training_usability", ""),
        "review_severity": row.get("review_severity", ""),
        "review_score": row.get("review_score", ""),
        "exclusion_reasons": row.get("policy_moderate_exclusion_reasons") or row.get("exclusion_reasons") or "",
        "quality_score": row.get("quality_score", ""),
        "mouth_activity_score": row.get("mouth_activity_score", ""),
        "mouth_visibility_score": row.get("mouth_visibility_score", ""),
        "scene_cut_score": row.get("scene_cut_score", ""),
        "blur_score": row.get("blur_score", ""),
    }


def valor_policy_moderate_v2(row: dict[str, Any]) -> str:
    return str(row.get("policy_moderate_v2") or row.get("policy_moderate") or "")


def clave_estable(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("source_id") or row.get("titulo") or ""),
        str(row.get("clip") or ""),
        str(row.get("path_roi") or row.get("npz") or ""),
    )


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return RAIZ_REPO / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar muestra estratificada para evaluacion VSR barata.")
    parser.add_argument("--policy-analysis", default=str(DEFAULT_POLICY_ANALYSIS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = leer_csv(args.policy_analysis)
    sample = generar_muestra_estratificada(rows, per_group=args.per_group, seed=args.seed)
    output = escribir_csv(sample, args.output)
    print(json.dumps({"output": str(output), "resumen": resumen_muestra(sample)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
