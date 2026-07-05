"""Helpers livianos para comparar experimentos VSR."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


RESULT_COLUMNS = [
    "experiment",
    "source_id",
    "clip",
    "split",
    "training_usability",
    "policy_moderate",
    "reference",
    "hypothesis",
    "wer",
    "cer",
]


def _normalizar_texto(texto: str) -> str:
    return " ".join(str(texto or "").strip().split())


def _distancia_edicion(referencia: list[str], hipotesis: list[str]) -> int:
    prev = list(range(len(hipotesis) + 1))
    for i, ref_token in enumerate(referencia, start=1):
        curr = [i]
        for j, hyp_token in enumerate(hipotesis, start=1):
            costo = 0 if ref_token == hyp_token else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + costo,
                )
            )
        prev = curr
    return prev[-1]


def error_rate(referencia: Iterable[str], hipotesis: Iterable[str]) -> float:
    ref = list(referencia)
    hyp = list(hipotesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _distancia_edicion(ref, hyp) / len(ref)


def wer(referencia: str, hipotesis: str) -> float:
    ref = _normalizar_texto(referencia).split()
    hyp = _normalizar_texto(hipotesis).split()
    return error_rate(ref, hyp)


def cer(referencia: str, hipotesis: str) -> float:
    ref = list(_normalizar_texto(referencia).replace(" ", ""))
    hyp = list(_normalizar_texto(hipotesis).replace(" ", ""))
    return error_rate(ref, hyp)


def cargar_resultados(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cargar_resultados_si_existen(path: str | Path) -> list[dict[str, str]]:
    archivo = Path(path)
    if not archivo.exists():
        return []
    return cargar_resultados(archivo)


def completar_metricas(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    completos: list[dict[str, str]] = []
    for row in rows:
        nuevo = dict(row)
        if not nuevo.get("wer"):
            nuevo["wer"] = f"{wer(nuevo.get('reference', ''), nuevo.get('hypothesis', '')):.6f}"
        if not nuevo.get("cer"):
            nuevo["cer"] = f"{cer(nuevo.get('reference', ''), nuevo.get('hypothesis', '')):.6f}"
        completos.append(nuevo)
    return completos


def _float(row: dict[str, str], col: str) -> float | None:
    valor = row.get(col, "")
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def resumen_por_grupo(
    rows: Iterable[dict[str, str]],
    group_col: str,
    min_clips: int = 30,
) -> list[dict[str, object]]:
    grupos: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grupos[row.get(group_col, "") or "(sin valor)"].append(row)

    salida: list[dict[str, object]] = []
    for grupo, filas in sorted(grupos.items()):
        wers = [v for v in (_float(row, "wer") for row in filas) if v is not None]
        cers = [v for v in (_float(row, "cer") for row in filas) if v is not None]
        n = len(filas)
        salida.append(
            {
                group_col: grupo,
                "clips": n,
                "wer": mean(wers) if wers else None,
                "cer": mean(cers) if cers else None,
                "can_conclude": n >= min_clips,
                "warning": "" if n >= min_clips else f"grupo chico: n={n} < {min_clips}",
            }
        )
    return salida


def comparar_experimentos(
    rows: Iterable[dict[str, str]],
    baseline: str = "baseline_original",
    candidato: str = "visual_cleaned",
    min_clips: int = 30,
) -> dict[str, object]:
    por_exp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        por_exp[row.get("experiment", "")].append(row)

    base = por_exp.get(baseline, [])
    cand = por_exp.get(candidato, [])
    base_wer = [v for v in (_float(row, "wer") for row in base) if v is not None]
    cand_wer = [v for v in (_float(row, "wer") for row in cand) if v is not None]
    base_cer = [v for v in (_float(row, "cer") for row in base) if v is not None]
    cand_cer = [v for v in (_float(row, "cer") for row in cand) if v is not None]
    can_conclude = len(base) >= min_clips and len(cand) >= min_clips

    return {
        "baseline": baseline,
        "candidate": candidato,
        "baseline_clips": len(base),
        "candidate_clips": len(cand),
        "baseline_wer": mean(base_wer) if base_wer else None,
        "candidate_wer": mean(cand_wer) if cand_wer else None,
        "delta_wer": (mean(cand_wer) - mean(base_wer)) if base_wer and cand_wer else None,
        "baseline_cer": mean(base_cer) if base_cer else None,
        "candidate_cer": mean(cand_cer) if cand_cer else None,
        "delta_cer": (mean(cand_cer) - mean(base_cer)) if base_cer and cand_cer else None,
        "can_conclude": can_conclude,
        "warning": "" if can_conclude else f"faltan clips suficientes: min_clips={min_clips}",
    }


def grupos_chicos(
    rows: Iterable[dict[str, str]],
    group_col: str,
    min_clips: int = 30,
) -> list[dict[str, object]]:
    return [
        row
        for row in resumen_por_grupo(rows, group_col, min_clips=min_clips)
        if not row["can_conclude"]
    ]
