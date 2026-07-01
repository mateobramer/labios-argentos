"""Analisis de politicas candidatas para la auditoria visual ROI.

Parte del manifest sanity y agrega columnas de severidad/retenion sin cambiar splits,
copiar datos ni entrenar modelos.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_cleaning.src.visual_quality_report import (
    cargar_metricas_clip_csv,
    cargar_predicciones_ref_hyp,
    repo_rel,
)


RAIZ_REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RAIZ_REPO / "data" / "metadata" / "visual_quality_manifest_full_roi_sanity.csv"
DEFAULT_OUTPUT = RAIZ_REPO / "data" / "metadata" / "visual_quality_policy_analysis_v2.csv"
DEFAULT_MODERATE_KEEP = RAIZ_REPO / "data" / "metadata" / "visual_quality_policy_moderate_keep_v2.csv"
DEFAULT_STRICT_KEEP = RAIZ_REPO / "data" / "metadata" / "visual_quality_policy_strict_keep_v2.csv"
DEFAULT_MAPEO = RAIZ_REPO / "evaluation" / "outputs" / "mapeo.csv"


POLICY_COLUMNS = [
    "review_severity",
    "review_score",
    "review_reason_group",
    "quality_percentile",
    "training_usability",
    "training_usability_reasons",
    "policy_conservative",
    "policy_conservative_exclusion_reasons",
    "policy_moderate",
    "policy_moderate_exclusion_reasons",
    "policy_strict",
    "policy_strict_exclusion_reasons",
]


GROUPS_BY_REASON = {
    "blur": "blur",
    "blur_extremo": "blur",
    "corte_escena": "scene_cut",
    "baja_textura_boca": "mouth_quality",
    "boca_inactiva": "mouth_quality",
    "boca_tapada_o_poco_visible": "mouth_quality",
    "demasiados_frames_boca_inactiva": "mouth_quality",
    "contraste_bajo": "contrast",
    "iluminacion_baja": "brightness",
    "movimiento_bajo": "motion",
    "frames_faltantes": "frames",
}

REVIEW_WEIGHTS = {
    "blur": 1.0,
    "contraste_bajo": 1.0,
    "iluminacion_baja": 1.0,
    "blur_extremo": 3.0,
    "corte_escena": 3.0,
    "baja_textura_boca": 2.0,
    "boca_inactiva": 2.0,
    "boca_tapada_o_poco_visible": 2.0,
    "demasiados_frames_boca_inactiva": 1.5,
    "movimiento_bajo": 2.0,
    "frames_faltantes": 2.0,
}

STRONG_COMBOS = [
    ("blur_extremo", "corte_escena"),
    ("blur_extremo", "baja_textura_boca"),
    ("blur_extremo", "contraste_bajo"),
    ("boca_inactiva", "baja_textura_boca"),
]

HARD_FAIL_USABILITY_REASONS = {
    "input_visual_missing",
    "video_no_legible",
    "sin_frames",
    "oscuridad_extrema",
    "freeze_extremo_confirmado",
    "boca_totalmente_no_visible_roi",
}


@dataclass(frozen=True)
class PolicyConfig:
    moderate_quality_pct: float = 10.0
    strict_quality_pct: float = 20.0
    min_retention_warning_pct: float = 70.0
    min_keep_clips_warning: int = 5


def leer_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def agregar_analisis_politicas(
    rows: list[dict[str, Any]],
    *,
    config: PolicyConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or PolicyConfig()
    thresholds = calcular_umbral_percentiles(rows, config)
    salida = []
    for row in rows:
        row_out = dict(row)
        razones_review = set(_split(row.get("review_reasons") or row.get("used_for_decision_reasons")))
        razones_hard = _split(row.get("hard_fail_reasons"))
        quality = _float(row.get("quality_score"))
        percentile = percentile_quality(quality, thresholds["qualities"])

        usability, usability_reasons = clasificar_training_usability(
            row,
            razones_review=razones_review,
            razones_hard=razones_hard,
            quality=quality,
            quality_percentile=percentile,
            moderate_quality_threshold=thresholds["moderate_quality_threshold"],
        )
        severity, review_score, groups = clasificar_review(
            razones_review,
            quality=quality,
            quality_percentile=percentile,
            moderate_quality_threshold=thresholds["moderate_quality_threshold"],
            usability=usability,
        )
        row_out["review_severity"] = severity
        row_out["review_score"] = f"{review_score:.2f}"
        row_out["review_reason_group"] = ";".join(groups) if groups else "none"
        row_out["quality_percentile"] = f"{percentile:.2f}"
        row_out["training_usability"] = usability
        row_out["training_usability_reasons"] = ";".join(usability_reasons)

        for policy_name in ("conservative", "moderate", "strict"):
            reasons = razones_exclusion_policy(
                policy_name,
                razones_review=razones_review,
                razones_hard=razones_hard,
                quality=quality,
                thresholds=thresholds,
                training_usability=usability,
                training_usability_reasons=usability_reasons,
            )
            row_out[f"policy_{policy_name}"] = "exclude" if reasons else "keep"
            row_out[f"policy_{policy_name}_exclusion_reasons"] = ";".join(reasons)
        salida.append(row_out)
    return salida


def calcular_umbral_percentiles(rows: list[dict[str, Any]], config: PolicyConfig) -> dict[str, Any]:
    qualities = sorted(_float(row.get("quality_score")) for row in rows if str(row.get("quality_score", "")) != "")
    return {
        "qualities": qualities,
        "moderate_quality_threshold": percentile_threshold(qualities, config.moderate_quality_pct),
        "strict_quality_threshold": percentile_threshold(qualities, config.strict_quality_pct),
        "moderate_quality_pct": config.moderate_quality_pct,
        "strict_quality_pct": config.strict_quality_pct,
    }


def clasificar_review(
    razones_review: set[str],
    *,
    quality: float,
    quality_percentile: float,
    moderate_quality_threshold: float,
    usability: str,
) -> tuple[str, float, list[str]]:
    score = sum(REVIEW_WEIGHTS.get(reason, 1.0) for reason in razones_review)
    groups = sorted({GROUPS_BY_REASON.get(reason, reason) for reason in razones_review})
    combo_fuerte = any(a in razones_review and b in razones_review for a, b in STRONG_COMBOS)
    if quality <= moderate_quality_threshold:
        score += 2.0
        groups.append("quality_tail")
    if combo_fuerte:
        score += 2.0
        groups.append("strong_combo")

    groups = sorted(dict.fromkeys(groups))
    if not razones_review and quality > moderate_quality_threshold:
        return "none", 0.0, []

    if usability == "bad_candidate" or combo_fuerte or quality_percentile <= 10.0 or score >= 5.0:
        severity = "high"
    elif {"corte_escena", "baja_textura_boca", "boca_inactiva"} & razones_review or score >= 3.0:
        severity = "medium"
    else:
        severity = "low"
    return severity, round(score, 2), groups


def clasificar_training_usability(
    row: dict[str, Any],
    *,
    razones_review: set[str],
    razones_hard: list[str],
    quality: float,
    quality_percentile: float,
    moderate_quality_threshold: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    hard_reales = [reason for reason in razones_hard if reason in HARD_FAIL_USABILITY_REASONS]
    if hard_reales:
        reasons.extend(f"hard_fail:{reason}" for reason in hard_reales)

    mouth_visibility = _float(row.get("mouth_visibility_score"))
    mouth_activity = _float(row.get("mouth_activity_score"))
    mouth_inactive = _float(row.get("mouth_inactive_frame_ratio"))
    scene_cut = _float(row.get("scene_cut_score"))
    blur = _float(row.get("blur_score"))
    mouth_texture = _float(row.get("mouth_texture_score"))

    mouth_visible = mouth_visibility >= 0.30 and mouth_activity >= 0.20 and mouth_inactive < 0.85
    if mouth_visible:
        reasons.append("mouth_visible")
    if mouth_visibility < 0.08:
        reasons.append("mouth_missing_many_frames")
    elif mouth_visibility < 0.30 or "boca_tapada_o_poco_visible" in razones_review:
        reasons.append("mouth_occluded_many_frames")
    if mouth_activity < 0.20 or "boca_inactiva" in razones_review or mouth_inactive >= 0.85:
        reasons.append("low_mouth_motion")
    if mouth_visibility < 0.22 and "corte_escena" in razones_review:
        reasons.append("roi_unstable_or_bad_crop")
    if mouth_visibility < 0.30 and scene_cut >= 0.72:
        reasons.append("side_profile_or_partial_mouth")
    has_blur_or_low_texture = bool({"blur", "blur_extremo", "baja_textura_boca"} & razones_review) or blur < 0.20 or mouth_texture < 0.25
    strong_blur_or_low_texture = "blur_extremo" in razones_review or "baja_textura_boca" in razones_review or mouth_texture < 0.25
    if has_blur_or_low_texture:
        reasons.append("blur_low_texture")
    if "corte_escena" in razones_review or scene_cut >= 0.72:
        reasons.append("scene_discontinuity")
    if quality <= moderate_quality_threshold or quality_percentile <= 10.0:
        reasons.append("quality_tail")

    reasons = list(dict.fromkeys(reasons))
    reason_set = set(reasons)

    bad = bool(hard_reales)
    bad = bad or "mouth_missing_many_frames" in reason_set
    bad = bad or ("mouth_occluded_many_frames" in reason_set and "low_mouth_motion" in reason_set)
    bad = bad or "roi_unstable_or_bad_crop" in reason_set
    bad = bad or ("low_mouth_motion" in reason_set and "blur_low_texture" in reason_set)
    bad = bad or (
        "scene_discontinuity" in reason_set
        and (
            (strong_blur_or_low_texture and "blur_low_texture" in reason_set)
            or bool({"low_mouth_motion", "mouth_occluded_many_frames", "roi_unstable_or_bad_crop"} & reason_set)
        )
    )
    bad = bad or (
        "blur_extremo" in razones_review
        and "baja_textura_boca" in razones_review
        and ("boca_inactiva" in razones_review or "low_mouth_motion" in reason_set)
    )

    if bad:
        usability = "bad_candidate"
    elif reason_set - {"mouth_visible"}:
        usability = "questionable"
    else:
        usability = "usable"
    return usability, reasons


def razones_exclusion_policy(
    policy_name: str,
    *,
    razones_review: set[str],
    razones_hard: list[str],
    quality: float,
    thresholds: dict[str, Any],
    training_usability: str,
    training_usability_reasons: list[str],
) -> list[str]:
    reasons = [f"hard_fail:{reason}" for reason in razones_hard]
    if policy_name == "conservative":
        return reasons

    if policy_name == "moderate":
        if training_usability == "bad_candidate":
            reasons.extend(f"training_usability:{reason}" for reason in training_usability_reasons if reason != "mouth_visible")
        return list(dict.fromkeys(reasons))

    if policy_name == "strict":
        if training_usability == "bad_candidate":
            reasons.extend(f"training_usability:{reason}" for reason in training_usability_reasons if reason != "mouth_visible")
        if "blur_extremo" in razones_review:
            reasons.append("blur_extremo")
        if "corte_escena" in razones_review:
            reasons.append("corte_escena")
        if "baja_textura_boca" in razones_review:
            reasons.append("baja_textura_boca")
        if quality <= thresholds["strict_quality_threshold"]:
            reasons.append(f"quality_worst_{thresholds['strict_quality_pct']:.0f}pct")
        return list(dict.fromkeys(reasons))

    raise ValueError(f"Politica desconocida: {policy_name}")


def resumen_politicas(rows: list[dict[str, Any]], *, config: PolicyConfig | None = None) -> dict[str, Any]:
    config = config or PolicyConfig()
    return {
        "clips": len(rows),
        "review_severity": dict(sorted(Counter(row.get("review_severity", "") for row in rows).items())),
        "training_usability": dict(sorted(Counter(row.get("training_usability", "") for row in rows).items())),
        "retencion": {
            policy: retencion_policy(rows, f"policy_{policy}")
            for policy in ("conservative", "moderate", "strict")
        },
        "policy_moderate_top_exclusion_reasons": dict(exclusion_reasons(rows, "policy_moderate").most_common(10)),
        "policy_strict_top_exclusion_reasons": dict(exclusion_reasons(rows, "policy_strict").most_common(10)),
        "policy_moderate_por_split": retencion_por(rows, "policy_moderate", "split"),
        "policy_moderate_por_fuente": retencion_por(rows, "policy_moderate", "source_id"),
        "alertas": {
            policy: alertas_retencion(rows, f"policy_{policy}", config=config)
            for policy in ("conservative", "moderate", "strict")
        },
    }


def retencion_policy(rows: list[dict[str, Any]], policy_col: str) -> dict[str, Any]:
    total = len(rows)
    keep = sum(1 for row in rows if row.get(policy_col) == "keep")
    exclude = total - keep
    return {
        "keep": keep,
        "exclude": exclude,
        "retention_pct": round(keep / max(total, 1) * 100, 2),
        "exclude_pct": round(exclude / max(total, 1) * 100, 2),
    }


def retencion_por(rows: list[dict[str, Any]], policy_col: str, group_col: str) -> list[dict[str, Any]]:
    grupos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grupos[str(row.get(group_col, ""))].append(row)
    salida = []
    for group, items in sorted(grupos.items()):
        ret = retencion_policy(items, policy_col)
        salida.append({group_col: group, "total": len(items), **ret})
    return salida


def alertas_retencion(rows: list[dict[str, Any]], policy_col: str, *, config: PolicyConfig) -> list[str]:
    alertas = []
    for group_col, label in (("split", "split"), ("source_id", "fuente")):
        for item in retencion_por(rows, policy_col, group_col):
            if item["retention_pct"] < config.min_retention_warning_pct:
                alertas.append(f"{label}_retencion_baja:{item[group_col]}={item['retention_pct']:.2f}%")
            if item["keep"] < config.min_keep_clips_warning and item["total"] > 0:
                alertas.append(f"{label}_pocos_keep:{item[group_col]}={item['keep']}")
    return alertas


def exclusion_reasons(rows: list[dict[str, Any]], policy_col: str) -> Counter[str]:
    reason_col = f"{policy_col}_exclusion_reasons"
    return Counter(reason for row in rows for reason in _split(row.get(reason_col)))


def escribir_candidatos(rows: list[dict[str, Any]], output_path: str | Path, *, policy_col: str) -> Path:
    keep_rows = [row for row in rows if row.get(policy_col) == "keep"]
    return escribir_csv(keep_rows, output_path, fieldnames=list(rows[0].keys()) if rows else [])


def buscar_predicciones_joinables() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    roots = [
        RAIZ_REPO / "vsr_models" / "runs" / "_shared",
        RAIZ_REPO / "vsr_models" / "runs",
        RAIZ_REPO / "evaluation" / "outputs",
        RAIZ_REPO / "data" / "metadata",
    ]
    for root in roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            if _csv_tiene_metricas_clip(csv_path):
                candidates.append({"tipo": "csv_metricas_clip", "path": repo_rel(csv_path)})
        for inf_path in sorted(root.rglob("*.inf")):
            if DEFAULT_MAPEO.exists():
                candidates.append({"tipo": "inf_ref_hyp_con_mapeo", "path": repo_rel(inf_path), "mapeo": repo_rel(DEFAULT_MAPEO)})
    return candidates


def cargar_predicciones_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    tipo = candidate.get("tipo")
    path = RAIZ_REPO / str(candidate.get("path", ""))
    if tipo == "csv_metricas_clip":
        return cargar_metricas_clip_csv(path)
    if tipo == "inf_ref_hyp_con_mapeo":
        return cargar_predicciones_ref_hyp(path, RAIZ_REPO / str(candidate.get("mapeo", "")))
    return []


def mejor_prediccion_joinable(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {(row.get("source_id", ""), row.get("clip", "")) for row in rows}
    mejor = {"candidate": None, "predicciones": [], "clips_matcheados": 0}
    for candidate in buscar_predicciones_joinables():
        try:
            predicciones = cargar_predicciones_candidate(candidate)
        except Exception as exc:  # pragma: no cover - defensivo ante archivos exploratorios
            candidate = dict(candidate)
            candidate["error"] = f"{type(exc).__name__}: {exc}"
            continue
        pred_keys = {(row.get("source_id", ""), row.get("clip", "")) for row in predicciones}
        matched = len(keys & pred_keys)
        if matched > mejor["clips_matcheados"]:
            mejor = {"candidate": candidate, "predicciones": predicciones, "clips_matcheados": matched}
    return mejor


def resumen_impacto_wer_cer(
    rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    min_clips: int = 30,
) -> dict[str, Any]:
    pred_por_clip = {(row.get("source_id", ""), row.get("clip", "")): row for row in pred_rows}
    joined = []
    for row in rows:
        pred = pred_por_clip.get((row.get("source_id", ""), row.get("clip", "")))
        if not pred:
            continue
        item = dict(row)
        item["wer_clip"] = _float(pred.get("wer_clip", pred.get("wer")))
        item["cer_clip"] = _float(pred.get("cer_clip", pred.get("cer")))
        joined.append(item)

    policy_moderate_counts = Counter(str(row.get("policy_moderate", "")) for row in joined)
    suficiente = len(joined) >= min_clips
    moderate_comparable = policy_moderate_counts.get("keep", 0) >= min_clips and policy_moderate_counts.get("exclude", 0) >= min_clips
    warnings = []
    if not suficiente:
        warnings.append("Menos de 30 clips matcheados: no sacar conclusiones de impacto.")
    if suficiente and policy_moderate_counts.get("exclude", 0) == 0:
        warnings.append(
            f"WER actual no valida policy_moderate porque los {len(joined)} clips matcheados estan todos en keep."
        )
    elif suficiente and not moderate_comparable:
        warnings.append("WER/CER no tiene al menos 30 clips por grupo keep/exclude de policy_moderate.")
    salida = {
        "clips_matcheados": len(joined),
        "min_clips_para_concluir": min_clips,
        "suficiente_para_concluir": suficiente,
        "policy_moderate_counts": dict(sorted(policy_moderate_counts.items())),
        "policy_moderate_comparable": moderate_comparable,
        "warning": " ".join(warnings),
        "por_decision": _metricas_por_grupo(joined, "decision"),
        "por_review_severity": _metricas_por_grupo(joined, "review_severity"),
        "por_training_usability": _metricas_por_grupo(joined, "training_usability"),
        "por_policy_conservative": _metricas_por_grupo(joined, "policy_conservative"),
        "por_policy_moderate": _metricas_por_grupo(joined, "policy_moderate"),
        "por_policy_strict": _metricas_por_grupo(joined, "policy_strict"),
        "por_review_reason_group": _metricas_por_grupo_multivalor(joined, "review_reason_group"),
    }
    return salida


def _metricas_por_grupo(rows: list[dict[str, Any]], group_col: str) -> list[dict[str, Any]]:
    grupos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grupos[str(row.get(group_col, ""))].append(row)
    return [_resumen_grupo(group_col, group, items) for group, items in sorted(grupos.items())]


def _metricas_por_grupo_multivalor(rows: list[dict[str, Any]], group_col: str) -> list[dict[str, Any]]:
    grupos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group in _split(row.get(group_col)) or ["none"]:
            grupos[group].append(row)
    salida = [_resumen_grupo(group_col, group, items) for group, items in sorted(grupos.items())]
    salida.sort(key=lambda item: item["clips"], reverse=True)
    return salida


def _resumen_grupo(group_col: str, group: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        group_col: group,
        "clips": len(items),
        "wer_promedio": round(float(np.mean([_float(item.get("wer_clip")) for item in items])), 4) if items else "",
        "cer_promedio": round(float(np.mean([_float(item.get("cer_clip")) for item in items])), 4) if items else "",
    }


def percentile_threshold(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = min(max(pct, 0.0), 100.0)
    index = int(np.ceil(len(values) * pct / 100.0)) - 1
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


def percentile_quality(value: float, sorted_values: list[float]) -> float:
    if not sorted_values:
        return 0.0
    count = sum(1 for item in sorted_values if item <= value)
    return round(count / len(sorted_values) * 100, 2)


def _csv_tiene_metricas_clip(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError):
        return False
    fields = set(header)
    return {"source_id", "clip", "wer", "cer"}.issubset(fields)


def _split(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera politicas candidatas desde el manifest visual sanity.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Manifest sanity de entrada.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV de salida con columnas de politicas.")
    parser.add_argument("--moderate-keep-output", default=str(DEFAULT_MODERATE_KEEP))
    parser.add_argument("--strict-keep-output", default=str(DEFAULT_STRICT_KEEP))
    parser.add_argument("--moderate-quality-pct", type=float, default=PolicyConfig.moderate_quality_pct)
    parser.add_argument("--strict-quality-pct", type=float, default=PolicyConfig.strict_quality_pct)
    parser.add_argument("--min-impact-clips", type=int, default=30)
    args = parser.parse_args()

    config = PolicyConfig(
        moderate_quality_pct=args.moderate_quality_pct,
        strict_quality_pct=args.strict_quality_pct,
    )
    rows = leer_csv(args.input)
    enriched = agregar_analisis_politicas(rows, config=config)
    fieldnames = list(rows[0].keys()) + [col for col in POLICY_COLUMNS if col not in rows[0]] if rows else POLICY_COLUMNS
    escribir_csv(enriched, args.output, fieldnames=fieldnames)
    escribir_candidatos(enriched, args.moderate_keep_output, policy_col="policy_moderate")
    escribir_candidatos(enriched, args.strict_keep_output, policy_col="policy_strict")

    best = mejor_prediccion_joinable(enriched)
    impact = resumen_impacto_wer_cer(enriched, best["predicciones"], min_clips=args.min_impact_clips)
    summary = {
        "outputs": {
            "policy_analysis": repo_rel(args.output),
            "moderate_keep": repo_rel(args.moderate_keep_output),
            "strict_keep": repo_rel(args.strict_keep_output),
        },
        **resumen_politicas(enriched, config=config),
        "wer_cer_candidate": best["candidate"],
        "wer_cer_clips_matcheados": best["clips_matcheados"],
        "wer_cer_suficiente_para_concluir": impact["suficiente_para_concluir"],
        "wer_cer_warning": impact["warning"],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
