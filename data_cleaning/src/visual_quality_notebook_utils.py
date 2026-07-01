"""Utilidades livianas para notebooks de auditoria visual."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from data_cleaning.src.visual_quality_eval_sample import DEFAULT_OUTPUT as DEFAULT_SAMPLE_PATH
from data_cleaning.src.visual_quality_policy_analysis import mejor_prediccion_joinable, resumen_impacto_wer_cer


RAIZ_REPO = Path(__file__).resolve().parents[2]
SANITY_PATH = RAIZ_REPO / "data" / "metadata" / "visual_quality_manifest_full_roi_sanity.csv"
POLICY_V1_PATH = RAIZ_REPO / "data" / "metadata" / "visual_quality_policy_analysis.csv"
POLICY_V2_PATH = RAIZ_REPO / "data" / "metadata" / "visual_quality_policy_analysis_v2.csv"
NUMERIC_COLUMNS = [
    "quality_score",
    "quality_percentile",
    "review_score",
    "mouth_activity_score",
    "mouth_visibility_score",
    "mouth_texture_score",
    "scene_cut_score",
    "blur_score",
]


def cargar_tablas_auditoria(
    *,
    sanity_path: str | Path = SANITY_PATH,
    policy_v2_path: str | Path = POLICY_V2_PATH,
    policy_v1_path: str | Path = POLICY_V1_PATH,
    sample_path: str | Path = DEFAULT_SAMPLE_PATH,
) -> dict[str, Any]:
    sanity = cargar_csv(sanity_path)
    policy = cargar_csv(policy_v2_path)
    policy_v1 = cargar_csv(policy_v1_path) if repo_path(policy_v1_path).exists() else None
    sample = cargar_csv(sample_path) if repo_path(sample_path).exists() else None
    return {
        "root": RAIZ_REPO,
        "sanity": sanity,
        "policy": policy,
        "policy_v1": policy_v1,
        "sample": sample,
        "sample_path": repo_path(sample_path),
    }


def cargar_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    return normalizar_numericas(frame)


def normalizar_numericas(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in NUMERIC_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def resumen_ejecutivo(sanity: pd.DataFrame, policy: pd.DataFrame, policy_v1: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = [
        {"metric": "clips", "value": len(sanity)},
        {"metric": "input_kind", "value": conteo_dict(sanity, "input_kind")},
        {"metric": "run_mode", "value": conteo_dict(sanity, "run_mode")},
        {"metric": "roi_coverage", "value": _roi_coverage(sanity)},
        {"metric": "decision_sanity", "value": conteo_dict(sanity, "decision")},
        {"metric": "review_severity_v2", "value": conteo_dict(policy, "review_severity")},
        {"metric": "training_usability", "value": conteo_dict(policy, "training_usability")},
        {"metric": "policy_conservative_v2", "value": retencion(policy, "policy_conservative")},
        {"metric": "policy_moderate_v2", "value": retencion(policy, "policy_moderate")},
        {"metric": "policy_strict_v2", "value": retencion(policy, "policy_strict")},
    ]
    if policy_v1 is not None:
        rows.append({"metric": "policy_moderate_v1", "value": retencion(policy_v1, "policy_moderate")})
    return pd.DataFrame(rows)


def tabla_retenciones(policy: pd.DataFrame, policy_v1: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = [
        {"version": "v2", **retencion(policy, "policy_conservative")},
        {"version": "v2", **retencion(policy, "policy_moderate")},
        {"version": "v2", **retencion(policy, "policy_strict")},
    ]
    if policy_v1 is not None and "policy_moderate" in policy_v1.columns:
        rows.append({"version": "v1", **retencion(policy_v1, "policy_moderate")})
    return pd.DataFrame(rows)


def retencion(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    counts = frame[col].value_counts().to_dict()
    keep = int(counts.get("keep", 0))
    exclude = int(counts.get("exclude", 0))
    total = len(frame)
    return {"policy": col, "keep": keep, "exclude": exclude, "retention_pct": round(keep / max(total, 1) * 100, 2)}


def conteo_dict(frame: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[col].value_counts().items()}


def top_reasons(frame: pd.DataFrame, col: str, *, top: int = 20) -> pd.DataFrame:
    if col not in frame.columns:
        return pd.DataFrame(columns=["reason", "clips"])
    counts = split_values(frame[col]).value_counts().head(top)
    return counts.rename_axis("reason").reset_index(name="clips")


def retencion_por_grupo(frame: pd.DataFrame, group_col: str, *, policy_col: str = "policy_moderate") -> pd.DataFrame:
    table = frame.groupby([group_col, policy_col]).size().unstack(fill_value=0)
    for col in ["keep", "exclude"]:
        if col not in table:
            table[col] = 0
    table["total"] = table["keep"] + table["exclude"]
    table["retention_pct"] = (table["keep"] / table["total"].clip(lower=1) * 100).round(2)
    return table.reset_index()[[group_col, "total", "keep", "exclude", "retention_pct"]]


def tabla_excluidos(policy: pd.DataFrame, *, n: int = 25) -> pd.DataFrame:
    cols = [
        "source_id",
        "clip",
        "policy_moderate",
        "training_usability",
        "training_usability_reasons",
        "policy_moderate_exclusion_reasons",
        "quality_score",
        "review_score",
        "review_reasons",
    ]
    return policy[policy.policy_moderate == "exclude"].sort_values(
        ["review_score", "quality_score"],
        ascending=[False, True],
    )[cols].head(n)


def candidatos_falsos_positivos(policy: pd.DataFrame, *, n: int = 20) -> pd.DataFrame:
    cols = [
        "source_id",
        "clip",
        "training_usability_reasons",
        "quality_score",
        "mouth_activity_score",
        "mouth_visibility_score",
        "scene_cut_score",
        "blur_score",
    ]
    candidatos = policy[
        (policy.policy_moderate == "exclude")
        & (policy.mouth_visibility_score >= 0.30)
        & (policy.mouth_activity_score >= 0.20)
    ].sort_values("quality_score", ascending=False)
    return candidatos[cols].head(n)


def analizar_wer_cer(policy: pd.DataFrame) -> dict[str, Any]:
    rows = policy.to_dict("records")
    best = mejor_prediccion_joinable(rows)
    impact = resumen_impacto_wer_cer(rows, best["predicciones"], min_clips=30)
    resumen = pd.DataFrame(
        [
            {
                "candidate": best["candidate"],
                "matched": impact["clips_matcheados"],
                "policy_moderate_counts": impact["policy_moderate_counts"],
                "comparable": impact["policy_moderate_comparable"],
                "warning": impact["warning"],
            }
        ]
    )
    grupos = {
        key: pd.DataFrame(impact[key])
        for key in [
            "por_decision",
            "por_review_severity",
            "por_training_usability",
            "por_policy_moderate",
            "por_policy_strict",
            "por_review_reason_group",
        ]
    }
    return {"best": best, "impact": impact, "resumen": resumen, "grupos": grupos}


def seleccionar_ejemplos_presentacion(policy: pd.DataFrame) -> pd.DataFrame:
    examples = []
    specs = [
        (policy[(policy.training_usability == "usable") & (policy.policy_moderate == "keep")], "quality_score", False),
        (policy[(policy.training_usability == "questionable") & (policy.policy_moderate == "keep")], "review_score", False),
        (policy[(policy.training_usability == "bad_candidate") & (policy.policy_moderate == "exclude")], "review_score", False),
    ]
    for subset, sort_col, ascending in specs:
        if not subset.empty:
            examples.append(subset.sort_values(sort_col, ascending=ascending).iloc[0])
    frontier = policy[
        (policy.training_usability == "questionable")
        & (policy.review_severity == "high")
        & (policy.policy_moderate == "keep")
    ]
    if frontier.empty:
        frontier = policy[(policy.policy_moderate == "keep") & (policy.review_severity != "none")]
    if not frontier.empty:
        examples.append(frontier.sort_values("quality_score").iloc[0])
    return pd.DataFrame([row.to_dict() for row in examples])


def generar_sheet_presentacion(policy: pd.DataFrame, preview_dir: str | Path) -> tuple[pd.DataFrame, Path | None]:
    examples = seleccionar_ejemplos_presentacion(policy)
    output = repo_path(preview_dir) / "presentacion_ejemplos.jpg"
    image = escribir_roi_contact_sheet(examples.to_dict("records"), output, label_mode="scores")
    return tabla_ejemplos(examples), image


def generar_sheets_diagnostico(policy: pd.DataFrame, preview_dir: str | Path) -> dict[str, Path | None]:
    preview_dir = repo_path(preview_dir)
    outputs: dict[str, Path | None] = {}
    reason_counts = top_reasons(
        policy.loc[policy.policy_moderate == "exclude"],
        "policy_moderate_exclusion_reasons",
        top=5,
    )
    for reason in reason_counts["reason"].tolist():
        subset = policy[
            (policy.policy_moderate == "exclude")
            & policy.policy_moderate_exclusion_reasons.astype(str).str.contains(reason, regex=False)
        ].sort_values("review_score", ascending=False).head(5)
        outputs[f"excluidos por {reason}"] = escribir_roi_contact_sheet(
            subset.to_dict("records"),
            preview_dir / f"reason_{nombre_seguro(reason)}.jpg",
        )
    false_positive_candidates = candidatos_falsos_positivos(policy, n=6)
    outputs["posibles falsos positivos"] = escribir_roi_contact_sheet(
        false_positive_candidates.to_dict("records"),
        preview_dir / "posibles_falsos_positivos.jpg",
    )
    frontier = policy[(policy.policy_moderate == "keep") & (policy.review_severity == "high")].sort_values("quality_score").head(6)
    outputs["fronterizos conservados"] = escribir_roi_contact_sheet(
        frontier.to_dict("records"),
        preview_dir / "fronterizos_conservados.jpg",
    )
    return outputs


def tabla_ejemplos(examples: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_id",
        "clip",
        "policy_moderate",
        "training_usability",
        "training_usability_reasons",
        "quality_score",
        "mouth_activity_score",
        "mouth_visibility_score",
        "scene_cut_score",
        "blur_score",
    ]
    return examples[[col for col in cols if col in examples.columns]]


def resumen_muestra_vsr(sample: pd.DataFrame | None) -> pd.DataFrame:
    if sample is None:
        return pd.DataFrame([{"metric": "visual_quality_vsr_eval_sample.csv", "value": "no generado"}])
    return pd.DataFrame(
        [
            {"metric": "clips", "value": len(sample)},
            {"metric": "policy_moderate_v2", "value": conteo_dict(sample, "policy_moderate_v2")},
            {"metric": "training_usability", "value": conteo_dict(sample, "training_usability")},
            {"metric": "split", "value": conteo_dict(sample, "split")},
            {"metric": "fuentes", "value": sample["source_id"].nunique() if "source_id" in sample.columns else 0},
        ]
    )


def composicion_muestra_vsr(sample: pd.DataFrame | None) -> pd.DataFrame:
    if sample is None:
        return pd.DataFrame()
    return sample.groupby(["policy_moderate_v2", "training_usability", "split"]).size().reset_index(name="clips")


def conclusion_presentacion(policy: pd.DataFrame, impact: dict[str, Any], sample: pd.DataFrame | None) -> str:
    moderate = retencion(policy, "policy_moderate")
    strict = retencion(policy, "policy_strict")
    msg = [
        "No entrenar todavia con un filtro visual final: falta medir impacto VSR barato por grupo.",
        f"policy_moderate_v2 retiene {moderate['retention_pct']:.2f}% ({moderate['keep']} clips) y excluye {moderate['exclude']} bad_candidate.",
        f"policy_strict_v2 retiene {strict['retention_pct']:.2f}% y queda solo como analisis de sensibilidad.",
    ]
    if impact["policy_moderate_comparable"]:
        msg.append("WER/CER tiene grupos comparables para evaluar policy_moderate_v2.")
    else:
        msg.append("WER/CER actual no valida excludes: la muestra estratificada es el siguiente insumo.")
    if sample is not None:
        msg.append(f"Muestra VSR disponible: {len(sample)} clips en data/metadata/visual_quality_vsr_eval_sample.csv.")
    return "  \n".join(msg)


def formato_predicciones_futuras() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"columna": "source_id", "descripcion": "fuente/titulo del manifest"},
            {"columna": "clip", "descripcion": "clip_NNNN original"},
            {"columna": "split", "descripcion": "split de la muestra"},
            {"columna": "reference", "descripcion": "texto esperado"},
            {"columna": "hypothesis", "descripcion": "salida VSR"},
            {"columna": "wer", "descripcion": "WER por clip, fraccion 0-1"},
            {"columna": "cer", "descripcion": "CER por clip, fraccion 0-1"},
        ]
    )


def comando_inferencia_existente() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "estado": "comando existente",
                "detalle": "vsr_main.py evalua un scenario exportado al layout Gimeno; no hay CLI directo para un CSV de muestra.",
            },
            {
                "estado": "inferencia",
                "detalle": "python vsr_main.py --database Rioplatense --scenario <scenario_muestra> --load-vsr $CKPT --output-dir ./spanish-benchmark/rioplatense/visual_quality_sample/",
            },
            {
                "estado": "cruce esperado",
                "detalle": "convertir inference/test.inf + mapeo.csv a CSV con source_id,clip,split,reference,hypothesis,wer,cer.",
            },
        ]
    )


def cargar_roi_frames(row: dict[str, Any], *, n: int = 5, size: int = 128) -> list[np.ndarray]:
    path = repo_path(row.get("path_roi", ""))
    if not path.exists() or path.suffix.lower() != ".npz":
        return []
    with np.load(path) as data:
        key = "rois" if "rois" in data.files else data.files[0]
        arr = data[key]
    if arr.ndim < 3 or arr.shape[0] == 0:
        return []
    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int)
    frames = arr[idx]
    if frames.ndim == 4 and frames.shape[-1] in (3, 4):
        gray = [cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY) for frame in frames]
    elif frames.ndim == 4 and frames.shape[1] in (3, 4):
        gray = [cv2.cvtColor(np.moveaxis(frame, 0, -1).astype(np.uint8), cv2.COLOR_BGR2GRAY) for frame in frames]
    else:
        gray = [frame[..., 0] if frame.ndim == 3 else frame for frame in frames]

    out = []
    for frame in gray:
        frame = np.asarray(frame)
        if frame.dtype.kind == "f" and frame.max(initial=0.0) <= 1.5:
            frame = frame * 255.0
        frame = cv2.resize(np.clip(frame, 0, 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        out.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
    return out


def escribir_roi_contact_sheet(
    rows: list[dict[str, Any]],
    output: str | Path,
    *,
    title: str = "",
    label_mode: str = "reasons",
) -> Path | None:
    rendered = []
    for row in rows:
        frames = cargar_roi_frames(row)
        if not frames:
            continue
        strip = np.concatenate(frames, axis=1)
        label = np.full((64, strip.shape[1], 3), 245, dtype=np.uint8)
        line1 = f"{row.get('clip','')} {row.get('policy_moderate','')} {row.get('training_usability','')} q={row.get('quality_score','')}"
        if label_mode == "scores":
            line2 = (
                f"act={row.get('mouth_activity_score','')} vis={row.get('mouth_visibility_score','')} "
                f"scene={row.get('scene_cut_score','')} blur={row.get('blur_score','')}"
            )
        else:
            line2 = str(row.get("training_usability_reasons") or row.get("policy_moderate_exclusion_reasons") or "")[:110]
        cv2.putText(label, line1[:120], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(label, line2[:120], (8, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        rendered.append(np.concatenate([label, strip], axis=0))
    if not rendered:
        return None
    width = max(item.shape[1] for item in rendered)
    normalized = []
    for item in rendered:
        if item.shape[1] < width:
            pad = np.full((item.shape[0], width - item.shape[1], 3), 245, dtype=np.uint8)
            item = np.concatenate([item, pad], axis=1)
        normalized.append(item)
    img = np.concatenate(normalized, axis=0)
    if title:
        title_bar = np.full((34, img.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(title_bar, title[:120], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        img = np.concatenate([title_bar, img], axis=0)
    output = repo_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return output if ok else None


def split_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(";").explode().replace("", pd.NA).dropna()


def repo_path(path: str | Path) -> Path:
    path = Path(str(path))
    if path.is_absolute():
        return path
    return RAIZ_REPO / path


def nombre_seguro(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return value[:80] or "item"


def _roi_coverage(sanity: pd.DataFrame) -> str:
    if "roi_coverage_at_run" in sanity.columns and len(sanity):
        value = pd.to_numeric(sanity["roi_coverage_at_run"], errors="coerce").dropna()
        if not value.empty:
            return f"{float(value.iloc[0]) * 100:.2f}%"
    roi = conteo_dict(sanity, "input_kind").get("roi_npz", 0)
    return f"{roi}/{len(sanity)} ({roi / max(len(sanity), 1) * 100:.2f}%)"
