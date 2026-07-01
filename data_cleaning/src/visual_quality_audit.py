"""Auditoria visual offline del dataset VSR.

Produce un manifest auditable con decision conservadora ``keep | review | drop``.
La auditoria no modifica datos originales, no toca splits y no entrena modelos.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_cleaning.src.visual_quality_metrics import (
    VideoFrames,
    cargar_frames,
    metricas_calidad_frames,
    metricas_rostro_haar,
)


RAIZ_REPO = Path(__file__).resolve().parents[2]
CLIPS_DIR = RAIZ_REPO / "data" / "clips"
ROIS_DIR = RAIZ_REPO / "data" / "processed" / "lip_rois"
DEFAULT_OUTPUT = RAIZ_REPO / "data" / "metadata" / "visual_quality_manifest_smoke.csv"
LIP_PREPROCESSING_MANIFEST = RAIZ_REPO / "data" / "metadata" / "lip_preprocessing_manifest.csv"
VERSION_AUDITORIA = "visual_quality_v4_roi_sanity"
ROI_DOWNLOAD_COMMAND = "gcloud storage rsync -r gs://labios-argentos-vsr-data/lip_rois ./data/processed/lip_rois"


ROI_INPUT_KINDS = {"roi_npz", "roi_video"}

ROI_DECISION_HARD_FAIL_REASONS = {
    "input_visual_missing",
    "video_no_legible",
    "sin_frames",
    "oscuridad_extrema",
    "freeze_extremo_confirmado",
    "boca_totalmente_no_visible_roi",
}

ROI_DECISION_REVIEW_REASONS = {
    "frames_faltantes",
    "iluminacion_baja",
    "blur_extremo",
    "blur",
    "movimiento_bajo",
    "boca_inactiva",
    "boca_tapada_o_poco_visible",
    "baja_textura_boca",
    "demasiados_frames_boca_inactiva",
    "contraste_bajo",
    "corte_escena",
}

FULL_FRAME_OR_HAAR_REASONS = {
    "face_detector_haar_alerta",
    "pose_proxy_haar",
    "multiples_caras",
    "rostro_inestable_o_no_detectado",
    "cambio_cara_dominante",
    "perfil_extremo",
    "face_tracking_proxy",
    "speaker_mismatch_proxy",
    "riesgo_boca_texto_o_hablante",
}

REASON_SCOPES = {
    **{reason: "roi_crop_metric" for reason in ROI_DECISION_HARD_FAIL_REASONS | ROI_DECISION_REVIEW_REASONS},
    **{reason: "full_frame_metric" for reason in FULL_FRAME_OR_HAAR_REASONS},
    "raw_fallback_low_confidence": "experimental_metric",
    "mouth_metrics_raw_clip_proxy": "experimental_metric",
}


ENCABEZADO = [
    "split",
    "spk",
    "source_id",
    "titulo",
    "clip",
    "path_roi",
    "path_video",
    "path_text",
    "input_kind",
    "audit_confidence",
    "audit_scope",
    "run_mode",
    "roi_coverage_at_run",
    "raw_fallback_count_at_run",
    "is_interpretable_for_vsr",
    "interpretation_warning",
    "n_frames",
    "frames_muestreados",
    "frames_esperados",
    "frame_read_ratio",
    "missing_frame_score",
    "face_detector",
    "frames_rostro_muestreados",
    "face_count_score",
    "track_stability_score",
    "ratio_cara",
    "avg_caras",
    "max_caras",
    "multi_face_risk",
    "dominant_face_shift",
    "face_box_area_jitter",
    "face_boxes_summary",
    "mouth_visibility_score",
    "mouth_activity_score",
    "mouth_texture_score",
    "mouth_inactive_frame_ratio",
    "pose_score",
    "pose_available",
    "pose_reason",
    "blur_score",
    "brightness_score",
    "contrast_score",
    "motion_score",
    "scene_cut_score",
    "scene_cut_count",
    "scene_cut_max_diff",
    "scene_cut_positions",
    "speaker_mismatch_risk",
    "speaker_mismatch_available",
    "speaker_mismatch_reason",
    "active_speaker_available",
    "active_speaker_reason",
    "quality_score",
    "quality_bucket",
    "decision",
    "decision_confidence",
    "metric_scope",
    "reasons",
    "used_for_decision_reasons",
    "hard_fail_reasons",
    "review_reasons",
    "experimental_reasons",
    "non_applicable_reasons",
    "invalid_for_input_reasons",
    "unavailable_signals",
    "version",
    "notes",
    "luma_media",
    "luma_p01",
    "luma_p99",
    "frac_oscuros",
    "contraste_medio",
    "blur_laplaciano",
    "movimiento_global",
    "actividad_boca",
    "textura_boca",
    "contraste_boca",
    "texto",
]


@dataclass(frozen=True)
class Thresholds:
    min_brightness_review: float = 0.30
    min_brightness_drop: float = 0.08
    min_blur_review: float = 0.20
    min_blur_drop: float = 0.04
    min_mouth_visibility_review: float = 0.30
    min_mouth_activity_review: float = 0.28
    min_motion_drop: float = 1.0
    min_frame_read_review: float = 0.90
    min_face_count_review: float = 0.35
    min_track_stability_review: float = 0.35
    min_pose_review: float = 0.30
    max_scene_cut_review: float = 0.72
    max_multi_face_review: float = 0.20
    max_speaker_mismatch_review: float = 0.55


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    decision_confidence: str
    quality_score: float
    quality_bucket: str
    reasons: list[str]
    used_for_decision_reasons: list[str]
    hard_fail_reasons: list[str]
    review_reasons: list[str]
    experimental_reasons: list[str]
    non_applicable_reasons: list[str]
    invalid_for_input_reasons: list[str]
    metric_scope: dict[str, str]
    unavailable_signals: list[str]


@dataclass(frozen=True)
class PreflightResult:
    total_rows: int
    rois_dir_exists: bool
    npz_count_on_disk: int
    roi_existing_rows: int
    roi_coverage: float
    raw_fallback_rows: int
    missing_visual_rows: int
    broken_roi_paths: int
    run_mode: str
    is_interpretable_for_vsr: bool
    interpretation_warning: str
    min_roi_coverage: float


def cargar_split(
    path: str | Path,
    *,
    source_id: str | None = None,
    limit: int | None = None,
    sample_strategy: str = "first",
    clips_per_source: int | None = None,
    max_clips: int | None = None,
    seed: int = 42,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            titulo = row.get("titulo") or row.get("source_id") or ""
            if source_id and titulo != source_id:
                continue
            row = dict(row)
            row["split_path"] = split_path.as_posix()
            rows.append(row)

    if sample_strategy == "stratified":
        rows = seleccionar_muestra_estratificada(
            rows,
            clips_per_source=clips_per_source,
            max_clips=max_clips or limit,
            seed=seed,
        )
    else:
        limite = max_clips or limit
        if limite is not None:
            rows = rows[:limite]
    return rows


def preflight_rois(rows: list[dict[str, str]], *, min_roi_coverage: float = 0.8) -> PreflightResult:
    total = len(rows)
    rois_dir_exists = ROIS_DIR.exists()
    npz_count = sum(1 for _ in ROIS_DIR.rglob("*.npz")) if rois_dir_exists else 0
    roi_existing = 0
    raw_fallback = 0
    missing_visual = 0
    broken_roi_paths = 0

    for row in rows:
        titulo = row.get("titulo") or row.get("source_id") or ""
        clip = row.get("clip") or ""
        paths = resolver_paths(row, titulo, clip)
        if paths["input_kind"] in {"roi_npz", "roi_video"}:
            roi_existing += 1
        elif paths["input_kind"] == "raw_clip":
            raw_fallback += 1
            if paths["path_roi"] and not Path(paths["path_roi"]).exists():
                broken_roi_paths += 1
        else:
            missing_visual += 1
            if paths["path_roi"] and not Path(paths["path_roi"]).exists():
                broken_roi_paths += 1

    coverage = roi_existing / total if total else 0.0
    raw_ratio = raw_fallback / total if total else 0.0
    if total == 0:
        run_mode = "smoke_raw_fallback"
        warning = "Split sin filas para auditar."
        interpretable = False
    elif raw_ratio > 0.5:
        run_mode = "smoke_raw_fallback"
        warning = (
            "Esta corrida NO audita el input real del VSR porque la mayoria cae a raw_clip. "
            "Sirve como smoke test de infraestructura, no para decidir filtros de entrenamiento."
        )
        interpretable = False
    elif coverage >= min_roi_coverage and raw_fallback == 0:
        run_mode = "roi_audit"
        warning = ""
        interpretable = True
    elif coverage >= min_roi_coverage:
        run_mode = "mixed_audit"
        warning = "Corrida mayormente sobre ROIs, pero con algunos fallbacks raw_clip; revisar esos casos antes de concluir."
        interpretable = True
    else:
        run_mode = "mixed_audit"
        warning = (
            "Cobertura ROI baja: la auditoria mezcla ROIs y fallbacks. "
            "No conviene usar retencion keep como filtro final."
        )
        interpretable = False

    return PreflightResult(
        total_rows=total,
        rois_dir_exists=rois_dir_exists,
        npz_count_on_disk=npz_count,
        roi_existing_rows=roi_existing,
        roi_coverage=round(coverage, 4),
        raw_fallback_rows=raw_fallback,
        missing_visual_rows=missing_visual,
        broken_roi_paths=broken_roi_paths,
        run_mode=run_mode,
        is_interpretable_for_vsr=interpretable,
        interpretation_warning=warning,
        min_roi_coverage=min_roi_coverage,
    )


def preflight_to_dict(preflight: PreflightResult) -> dict[str, Any]:
    return {
        "total_rows": preflight.total_rows,
        "rois_dir_exists": preflight.rois_dir_exists,
        "npz_count_on_disk": preflight.npz_count_on_disk,
        "roi_existing_rows": preflight.roi_existing_rows,
        "roi_coverage": preflight.roi_coverage,
        "raw_fallback_rows": preflight.raw_fallback_rows,
        "missing_visual_rows": preflight.missing_visual_rows,
        "broken_roi_paths": preflight.broken_roi_paths,
        "run_mode": preflight.run_mode,
        "is_interpretable_for_vsr": preflight.is_interpretable_for_vsr,
        "interpretation_warning": preflight.interpretation_warning,
        "min_roi_coverage": preflight.min_roi_coverage,
    }


def imprimir_preflight(preflight: PreflightResult, *, allow_raw_fallback: bool = False) -> None:
    print(
        f"ROI coverage: {preflight.roi_existing_rows}/{preflight.total_rows} clips "
        f"({preflight.roi_coverage * 100:.1f}%)"
    )
    print(f"ROIs dir exists: {preflight.rois_dir_exists} | .npz on disk: {preflight.npz_count_on_disk}")
    print(f"Raw fallback count: {preflight.raw_fallback_rows}")
    print(f"Missing visual inputs: {preflight.missing_visual_rows}")
    print(f"Broken ROI paths: {preflight.broken_roi_paths}")
    print(f"Run mode: {preflight.run_mode}")
    print(f"Interpretable for VSR: {preflight.is_interpretable_for_vsr}")
    if preflight.interpretation_warning:
        print(f"WARNING: {preflight.interpretation_warning}")
    if preflight.raw_fallback_rows and not allow_raw_fallback:
        print("WARNING: raw_clip fallback is being used without --allow-raw-fallback.")
    if preflight.roi_coverage < preflight.min_roi_coverage:
        print("For real VSR audit, download ROIs:")
        print(ROI_DOWNLOAD_COMMAND)


def validar_preflight(preflight: PreflightResult, *, require_roi: bool) -> None:
    if require_roi and preflight.roi_coverage < preflight.min_roi_coverage:
        raise RuntimeError(
            "ROI coverage below required threshold: "
            f"{preflight.roi_coverage:.1%} < {preflight.min_roi_coverage:.1%}. "
            f"Download ROIs first with: {ROI_DOWNLOAD_COMMAND}"
        )


def seleccionar_muestra_estratificada(
    rows: list[dict[str, str]],
    *,
    clips_per_source: int | None,
    max_clips: int | None,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grupos: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grupos[(row.get("split", ""), row.get("titulo") or row.get("source_id") or "")].append(row)

    seleccion: list[dict[str, str]] = []
    for key in sorted(grupos):
        grupo = list(grupos[key])
        grupo.sort(key=lambda r: (_clip_number(r.get("clip", "")), r.get("clip", "")))
        if clips_per_source is not None and len(grupo) > clips_per_source:
            indices = sorted(rng.sample(range(len(grupo)), clips_per_source))
            grupo = [grupo[i] for i in indices]
        seleccion.extend(grupo)

    if max_clips is not None and len(seleccion) > max_clips:
        seleccion = rng.sample(seleccion, max_clips)
    seleccion.sort(key=lambda r: (r.get("split", ""), r.get("titulo", ""), _clip_number(r.get("clip", ""))))
    return seleccion


def cargar_lip_preprocessing_manifest(path: str | Path = LIP_PREPROCESSING_MANIFEST) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            titulo = row.get("titulo") or ""
            clip = row.get("clip") or ""
            if titulo and clip:
                out[(titulo, clip)] = row
    return out


def auditar_split(
    *,
    split_path: str | Path,
    output_path: str | Path,
    source_id: str | None = None,
    limit: int | None = None,
    sample_strategy: str = "first",
    clips_per_source: int | None = None,
    max_clips: int | None = None,
    seed: int = 42,
    max_frames: int = 64,
    face_frames: int = 4,
    usar_haar: bool = True,
    thresholds: Thresholds | None = None,
    min_roi_coverage: float = 0.8,
    require_roi: bool = False,
) -> list[dict[str, Any]]:
    thresholds = thresholds or Thresholds()
    rows = cargar_split(
        split_path,
        source_id=source_id,
        limit=limit,
        sample_strategy=sample_strategy,
        clips_per_source=clips_per_source,
        max_clips=max_clips,
        seed=seed,
    )
    preflight = preflight_rois(rows, min_roi_coverage=min_roi_coverage)
    validar_preflight(preflight, require_roi=require_roi)
    lip_manifest = cargar_lip_preprocessing_manifest()
    auditadas = [
        auditar_fila(
            row,
            max_frames=max_frames,
            face_frames=face_frames,
            usar_haar=usar_haar,
            lip_manifest=lip_manifest,
            thresholds=thresholds,
            preflight=preflight,
        )
        for row in rows
    ]
    guardar_manifest(auditadas, output_path)
    return auditadas


def auditar_fila(
    row: dict[str, str],
    *,
    max_frames: int,
    face_frames: int,
    usar_haar: bool,
    lip_manifest: dict[tuple[str, str], dict[str, str]],
    thresholds: Thresholds,
    preflight: PreflightResult | None = None,
) -> dict[str, Any]:
    titulo = row.get("titulo") or row.get("source_id") or ""
    clip = row.get("clip") or ""
    split = row.get("split") or Path(row.get("split_path", "")).stem
    spk = row.get("spk") or row.get("speaker") or ""
    texto = row.get("texto") or leer_texto(CLIPS_DIR / titulo / f"{clip}.txt")

    paths = resolver_paths(row, titulo, clip)
    notes: list[str] = []
    if paths["input_kind"] == "missing":
        video = VideoFrames(
            frames=np.zeros((0, 0, 0), dtype=np.float32),
            n_frames_total=int(row.get("n_frames") or 0),
            fps=0.0,
            input_kind="missing",
            path=None,
        )
        calidad = metricas_calidad_frames(video)
        notes.append("input_visual_no_encontrado")
    else:
        try:
            video = cargar_frames(paths["input_path"], max_frames=max_frames, input_kind=paths["input_kind"])
            calidad = metricas_calidad_frames(video)
        except Exception as exc:  # pragma: no cover - defensivo ante codecs rotos
            notes.append(f"entrada_no_legible:{type(exc).__name__}")
            video = VideoFrames(
                frames=np.zeros((0, 0, 0), dtype=np.float32),
                n_frames_total=int(row.get("n_frames") or 0),
                fps=0.0,
                input_kind=paths["input_kind"],
                path=paths.get("input_path"),
            )
            calidad = metricas_calidad_frames(video)

    input_meta = metadata_input(paths["input_kind"])
    if paths["input_kind"] in ROI_INPUT_KINDS:
        rostro = rostro_no_disponible("not_applicable_roi_crop_sin_full_frame")
    elif usar_haar and paths["path_video"].exists():
        rostro = metricas_rostro_haar(paths["path_video"], max_frames=face_frames)
    else:
        rostro = rostro_no_disponible("sin_video_crudo_o_detector_desactivado")

    if paths["input_kind"] not in ROI_INPUT_KINDS:
        rostro = combinar_rostro_con_preproc(rostro, lip_manifest.get((titulo, clip)), notes)
    speaker = estimar_speaker_mismatch(calidad, rostro, texto, input_kind=paths["input_kind"])
    decision = decidir(
        calidad,
        rostro,
        texto,
        thresholds,
        input_kind=paths["input_kind"],
        audit_confidence=input_meta["audit_confidence"],
        speaker_mismatch_risk=speaker["speaker_mismatch_risk"],
    )
    if paths["input_kind"] == "raw_clip":
        notes.append("metricas_boca_sobre_clip_crudo_proxy")
    if rostro.get("face_notes"):
        notes.append(str(rostro["face_notes"]))

    row_out = {
        "split": split,
        "spk": spk,
        "source_id": titulo,
        "titulo": titulo,
        "clip": clip,
        "path_roi": repo_rel(paths["path_roi"]),
        "path_video": repo_rel(paths["path_video"]),
        "path_text": repo_rel(paths["path_text"]),
        "input_kind": paths["input_kind"],
        "audit_confidence": input_meta["audit_confidence"],
        "audit_scope": input_meta["audit_scope"],
        "run_mode": preflight.run_mode if preflight else "",
        "roi_coverage_at_run": preflight.roi_coverage if preflight else "",
        "raw_fallback_count_at_run": preflight.raw_fallback_rows if preflight else "",
        "is_interpretable_for_vsr": preflight.is_interpretable_for_vsr if preflight else "",
        "interpretation_warning": preflight.interpretation_warning if preflight else "",
        "n_frames": calidad["n_frames"] or row.get("n_frames", ""),
        "frames_muestreados": calidad["frames_muestreados"],
        "frames_esperados": calidad["frames_esperados"],
        "frame_read_ratio": calidad["frame_read_ratio"],
        "missing_frame_score": calidad["missing_frame_score"],
        "face_detector": rostro["face_detector"],
        "frames_rostro_muestreados": rostro["frames_rostro_muestreados"],
        "face_count_score": rostro["face_count_score"],
        "track_stability_score": rostro["track_stability_score"],
        "ratio_cara": rostro["ratio_cara"],
        "avg_caras": rostro["avg_caras"],
        "max_caras": rostro["max_caras"],
        "multi_face_risk": rostro["multi_face_risk"],
        "dominant_face_shift": rostro["dominant_face_shift"],
        "face_box_area_jitter": rostro["face_box_area_jitter"],
        "face_boxes_summary": rostro["face_boxes_summary"],
        "mouth_visibility_score": calidad["mouth_visibility_score"],
        "mouth_activity_score": calidad["mouth_activity_score"],
        "mouth_texture_score": calidad["mouth_texture_score"],
        "mouth_inactive_frame_ratio": calidad["mouth_inactive_frame_ratio"],
        "pose_score": rostro["pose_score"],
        "pose_available": rostro["pose_available"],
        "pose_reason": rostro["pose_reason"],
        "blur_score": calidad["blur_score"],
        "brightness_score": calidad["brightness_score"],
        "contrast_score": calidad["contrast_score"],
        "motion_score": calidad["motion_score"],
        "scene_cut_score": calidad["scene_cut_score"],
        "scene_cut_count": calidad["scene_cut_count"],
        "scene_cut_max_diff": calidad["scene_cut_max_diff"],
        "scene_cut_positions": calidad["scene_cut_positions"],
        "speaker_mismatch_risk": speaker["speaker_mismatch_risk"],
        "speaker_mismatch_available": speaker["speaker_mismatch_available"],
        "speaker_mismatch_reason": speaker["speaker_mismatch_reason"],
        "active_speaker_available": False,
        "active_speaker_reason": "not_available_sin_modelo_ligero_confiable",
        "quality_score": decision.quality_score,
        "quality_bucket": decision.quality_bucket,
        "decision": decision.decision,
        "decision_confidence": decision.decision_confidence,
        "metric_scope": formatear_metric_scope(decision.metric_scope),
        "reasons": ";".join(decision.reasons),
        "used_for_decision_reasons": ";".join(decision.used_for_decision_reasons),
        "hard_fail_reasons": ";".join(decision.hard_fail_reasons),
        "review_reasons": ";".join(decision.review_reasons),
        "experimental_reasons": ";".join(decision.experimental_reasons),
        "non_applicable_reasons": ";".join(decision.non_applicable_reasons),
        "invalid_for_input_reasons": ";".join(decision.invalid_for_input_reasons),
        "unavailable_signals": ";".join(decision.unavailable_signals),
        "version": VERSION_AUDITORIA,
        "notes": ";".join(dict.fromkeys(n for n in notes if n)),
        "luma_media": calidad["luma_media"],
        "luma_p01": calidad["luma_p01"],
        "luma_p99": calidad["luma_p99"],
        "frac_oscuros": calidad["frac_oscuros"],
        "contraste_medio": calidad["contraste_medio"],
        "blur_laplaciano": calidad["blur_laplaciano"],
        "movimiento_global": calidad["movimiento_global"],
        "actividad_boca": calidad["actividad_boca"],
        "textura_boca": calidad["textura_boca"],
        "contraste_boca": calidad["contraste_boca"],
        "texto": texto,
    }
    return row_out


def resolver_paths(row: dict[str, str], titulo: str, clip: str) -> dict[str, Any]:
    path_roi_row = _repo_path_or_none(row.get("path_roi"))
    path_npz = _repo_path_or_none(row.get("npz"))
    path_roi_mp4 = ROIS_DIR / titulo / f"{clip}.mp4"
    path_video = CLIPS_DIR / titulo / f"{clip}.mp4"
    path_text = CLIPS_DIR / titulo / f"{clip}.txt"

    for candidate in [path_roi_row, path_npz]:
        if candidate and candidate.exists() and candidate.suffix.lower() == ".npz":
            return _paths(candidate, "roi_npz", candidate, path_video, path_text)
    for candidate in [path_roi_row, path_roi_mp4]:
        if candidate and candidate.exists():
            return _paths(candidate, "roi_video", candidate, path_video, path_text)
    if path_video.exists():
        path_roi = path_npz or path_roi_row or path_roi_mp4
        return _paths(path_video, "raw_clip", path_roi, path_video, path_text)
    path_roi = path_npz or path_roi_row or path_roi_mp4
    return _paths(None, "missing", path_roi, path_video, path_text)


def metadata_input(input_kind: str) -> dict[str, str]:
    if input_kind == "roi_npz":
        return {"audit_confidence": "high", "audit_scope": "mide_exactamente_el_input_npz_del_vsr"}
    if input_kind == "roi_video":
        return {"audit_confidence": "medium", "audit_scope": "mide_roi_video_derivado_no_npz"}
    if input_kind == "raw_clip":
        return {"audit_confidence": "low", "audit_scope": "fallback_clip_crudo_no_equivale_al_roi_del_vsr"}
    return {"audit_confidence": "low", "audit_scope": "sin_input_visual_disponible"}


def decidir(
    calidad: dict[str, Any],
    rostro: dict[str, Any],
    texto: str,
    thresholds: Thresholds,
    *,
    input_kind: str = "roi_npz",
    audit_confidence: str = "high",
    speaker_mismatch_risk: float = 0.0,
) -> DecisionResult:
    hard_fail: list[str] = []
    review: list[str] = []
    experimental: list[str] = []
    non_applicable: list[str] = []
    invalid_for_input: list[str] = []
    unavailable = ["active_speaker_detection"]

    brightness = _float(calidad.get("brightness_score"))
    blur = _float(calidad.get("blur_score"))
    contrast = _float(calidad.get("contrast_score"))
    motion = _float(calidad.get("movimiento_global"))
    motion_score = _float(calidad.get("motion_score"))
    mouth_visibility = _float(calidad.get("mouth_visibility_score"))
    mouth_activity = _float(calidad.get("mouth_activity_score"))
    mouth_texture = _float(calidad.get("mouth_texture_score"))
    mouth_inactive = _float(calidad.get("mouth_inactive_frame_ratio"))
    scene_cut = _float(calidad.get("scene_cut_score"))
    frame_read = _float(calidad.get("frame_read_ratio"))
    n_frames = int(_float(calidad.get("n_frames")))
    face_count = _float(rostro.get("face_count_score"))
    track_stability = _float(rostro.get("track_stability_score"))
    pose = _float(rostro.get("pose_score"))
    pose_available = _as_bool(rostro.get("pose_available"))
    multi_face = _float(rostro.get("multi_face_risk"))
    frac_oscuros = _float(calidad.get("frac_oscuros"))
    roi_confiable = input_kind in ROI_INPUT_KINDS

    if roi_confiable:
        non_applicable.extend(
            [
                "face_detector_haar_alerta",
                "pose_proxy_haar",
                "multiples_caras",
                "rostro_inestable_o_no_detectado",
                "cambio_cara_dominante",
                "perfil_extremo",
                "face_tracking_proxy",
                "speaker_mismatch_proxy",
                "riesgo_boca_texto_o_hablante",
            ]
        )
        invalid_for_input.extend(non_applicable)
        unavailable.extend(
            [
                "haar_face_detector_roi_crop",
                "pose_proxy_haar_roi_crop",
                "multi_face_roi_crop",
                "speaker_mismatch_proxy_roi_crop",
            ]
        )

    if input_kind == "missing":
        hard_fail.append("input_visual_missing")
    if not calidad.get("video_legible", False):
        hard_fail.append("video_no_legible")
    if n_frames <= 0:
        hard_fail.append("sin_frames")
    if frame_read and frame_read < thresholds.min_frame_read_review:
        review.append("frames_faltantes")

    oscuridad_extrema = brightness < thresholds.min_brightness_drop or frac_oscuros > 0.80
    if oscuridad_extrema:
        hard_fail.append("oscuridad_extrema")
    elif brightness < thresholds.min_brightness_review:
        review.append("iluminacion_baja")

    if blur < thresholds.min_blur_drop and roi_confiable:
        review.append("blur_extremo")
    elif blur < thresholds.min_blur_review:
        review.append("blur")

    freeze_extremo_umbral = max(0.15, thresholds.min_motion_drop * 0.20)
    if motion <= freeze_extremo_umbral:
        if roi_confiable:
            hard_fail.append("freeze_extremo_confirmado")
        else:
            review.append("freeze_posible_raw_clip")
    elif motion_score < 0.20:
        review.append("movimiento_bajo")

    boca_totalmente_no_visible = mouth_visibility < 0.06 and mouth_activity < 0.06
    if texto.strip() and boca_totalmente_no_visible and roi_confiable:
        hard_fail.append("boca_totalmente_no_visible_roi")
    else:
        if texto.strip() and mouth_activity < thresholds.min_mouth_activity_review:
            review.append("boca_inactiva")
        if mouth_visibility < thresholds.min_mouth_visibility_review:
            review.append("boca_tapada_o_poco_visible")
        if mouth_texture < 0.25:
            review.append("baja_textura_boca")
        if mouth_inactive > 0.85 and texto.strip():
            review.append("demasiados_frames_boca_inactiva")

    if contrast < 0.25:
        review.append("contraste_bajo")
    if scene_cut > thresholds.max_scene_cut_review:
        review.append("corte_escena")

    if not roi_confiable:
        if multi_face > thresholds.max_multi_face_review:
            review.append("multiples_caras")
        if face_count < thresholds.min_face_count_review:
            review.append("rostro_inestable_o_no_detectado")
        if track_stability < thresholds.min_track_stability_review:
            review.append("cambio_cara_dominante")
        if pose_available and pose < thresholds.min_pose_review:
            review.append("perfil_extremo")
    if not pose_available:
        unavailable.append("pose_landmarks")

    if not roi_confiable and speaker_mismatch_risk > thresholds.max_speaker_mismatch_review:
        review.append("riesgo_boca_texto_o_hablante")

    if input_kind == "raw_clip":
        review.append("raw_fallback_low_confidence")
        experimental.append("mouth_metrics_raw_clip_proxy")
    if rostro.get("face_detector") == "haar":
        experimental.append("face_detector_haar_alerta")
        experimental.append("pose_proxy_haar")
    if speaker_mismatch_risk > 0:
        experimental.append("speaker_mismatch_proxy")

    quality_score = calcular_quality_score(
        brightness=brightness,
        blur=blur,
        contrast=contrast,
        mouth_visibility=mouth_visibility,
        mouth_activity=mouth_activity,
        motion_score=motion_score,
        scene_cut=scene_cut,
        face_count=face_count,
        track_stability=track_stability,
        pose=pose,
        multi_face=multi_face,
        frame_read=frame_read or 1.0,
        speaker_mismatch=speaker_mismatch_risk,
        input_kind=input_kind,
    )

    if hard_fail:
        decision = "drop"
        quality_score = min(quality_score, 0.35)
    elif review:
        decision = "review"
    else:
        decision = "keep"

    quality_bucket = bucket_quality(quality_score)
    decision_confidence = confianza_decision(decision, audit_confidence, hard_fail, review, experimental)
    used_for_decision = list(dict.fromkeys(hard_fail + review))
    reasons = used_for_decision
    non_applicable = list(dict.fromkeys(non_applicable))
    invalid_for_input = list(dict.fromkeys(invalid_for_input))
    metric_scope = metric_scope_por_reason(used_for_decision, experimental, non_applicable)
    return DecisionResult(
        decision=decision,
        decision_confidence=decision_confidence,
        quality_score=round(quality_score, 4),
        quality_bucket=quality_bucket,
        reasons=reasons,
        used_for_decision_reasons=used_for_decision,
        hard_fail_reasons=list(dict.fromkeys(hard_fail)),
        review_reasons=list(dict.fromkeys(review)),
        experimental_reasons=list(dict.fromkeys(experimental)),
        non_applicable_reasons=non_applicable,
        invalid_for_input_reasons=invalid_for_input,
        metric_scope=metric_scope,
        unavailable_signals=list(dict.fromkeys(unavailable)),
    )


def calcular_quality_score(**scores: Any) -> float:
    if scores.get("input_kind") in ROI_INPUT_KINDS:
        score = (
            0.16 * scores["brightness"]
            + 0.14 * scores["blur"]
            + 0.12 * scores["contrast"]
            + 0.18 * scores["mouth_visibility"]
            + 0.16 * scores["mouth_activity"]
            + 0.10 * scores["motion_score"]
            + 0.08 * scores["frame_read"]
            + 0.06 * (1.0 - scores["scene_cut"])
        )
        return float(np.clip(score, 0.0, 1.0))

    score = (
        0.14 * scores["brightness"]
        + 0.09 * scores["blur"]
        + 0.16 * scores["mouth_visibility"]
        + 0.14 * scores["mouth_activity"]
        + 0.10 * scores["motion_score"]
        + 0.09 * scores["face_count"]
        + 0.08 * scores["track_stability"]
        + 0.05 * scores["pose"]
        + 0.07 * scores["frame_read"]
        + 0.04 * (1.0 - scores["scene_cut"])
        + 0.02 * (1.0 - scores["multi_face"])
        + 0.02 * (1.0 - scores["speaker_mismatch"])
    )
    return float(np.clip(score, 0.0, 1.0))


def bucket_quality(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    if score >= 0.40:
        return "low"
    return "critical"


def metric_scope_por_reason(
    used_for_decision: list[str],
    experimental: list[str],
    non_applicable: list[str],
) -> dict[str, str]:
    scopes: dict[str, str] = {}
    for reason in used_for_decision:
        scopes[reason] = REASON_SCOPES.get(reason, "roi_crop_metric")
    for reason in experimental:
        scopes[reason] = "experimental_metric"
    for reason in non_applicable:
        scopes[reason] = "not_applicable"
    return scopes


def formatear_metric_scope(scopes: dict[str, str]) -> str:
    return ";".join(f"{reason}:{scope}" for reason, scope in sorted(scopes.items()))


def confianza_decision(
    decision: str,
    audit_confidence: str,
    hard_fail: list[str],
    review: list[str],
    experimental: list[str],
) -> str:
    if decision == "drop" and hard_fail and audit_confidence in {"high", "medium"}:
        return "high"
    if audit_confidence == "low":
        return "low"
    if decision == "review" and review and len(experimental) >= len(review):
        return "medium"
    return "medium" if decision == "review" else "high"


def estimar_speaker_mismatch(
    calidad: dict[str, Any],
    rostro: dict[str, Any],
    texto: str,
    *,
    input_kind: str,
) -> dict[str, Any]:
    if input_kind in ROI_INPUT_KINDS:
        return {
            "speaker_mismatch_risk": 0.0,
            "speaker_mismatch_available": False,
            "speaker_mismatch_reason": "not_applicable_roi_crop_sin_full_frame_active_speaker",
        }
    if not texto.strip():
        return {
            "speaker_mismatch_risk": 0.0,
            "speaker_mismatch_available": True,
            "speaker_mismatch_reason": "sin_texto_no_se_evalua_habla",
        }
    factores: list[tuple[str, float]] = []
    mouth_activity = _float(calidad.get("mouth_activity_score"))
    scene_cut = _float(calidad.get("scene_cut_score"))
    multi_face = _float(rostro.get("multi_face_risk"))
    track = _float(rostro.get("track_stability_score"))
    if mouth_activity < 0.25:
        factores.append(("texto_con_boca_poco_activa", 0.35))
    if multi_face > 0.20:
        factores.append(("multiples_caras_sin_active_speaker", 0.30))
    if scene_cut > 0.72:
        factores.append(("corte_visual_en_clip", 0.25))
    if track < 0.35:
        factores.append(("cara_dominante_inestable", 0.25))
    riesgo = min(1.0, sum(peso for _, peso in factores))
    reason = "|".join(nombre for nombre, _ in factores) if factores else "proxy_sin_alertas"
    return {
        "speaker_mismatch_risk": round(riesgo, 4),
        "speaker_mismatch_available": True,
        "speaker_mismatch_reason": reason,
    }


def combinar_rostro_con_preproc(
    rostro: dict[str, Any],
    preproc_row: dict[str, str] | None,
    notes: list[str],
) -> dict[str, Any]:
    if not preproc_row:
        return rostro
    try:
        ratio = float(preproc_row.get("ratio_deteccion") or 0.0)
    except ValueError:
        ratio = 0.0
    estado = preproc_row.get("estado") or ""
    if ratio > float(rostro.get("face_count_score") or 0.0):
        rostro = dict(rostro)
        rostro["face_count_score"] = round(ratio, 4)
        rostro["track_stability_score"] = round(max(float(rostro.get("track_stability_score") or 0.0), ratio), 4)
        notes.append("rostro_reforzado_por_lip_preprocessing_manifest")
    if estado and estado != "ok":
        notes.append(f"lip_preprocessing_estado:{estado}")
    return rostro


def guardar_manifest(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ENCABEZADO, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resumen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisiones = Counter(row["decision"] for row in rows)
    razones = contar_reasons(rows, "used_for_decision_reasons", fallback_columna="reasons")
    hard_fails = contar_reasons(rows, "hard_fail_reasons")
    review_reasons = contar_reasons(rows, "review_reasons")
    experimental = contar_reasons(rows, "experimental_reasons")
    non_applicable = contar_reasons(rows, "non_applicable_reasons")
    keep = decisiones.get("keep", 0)
    review = decisiones.get("review", 0)
    drop = decisiones.get("drop", 0)
    total = len(rows) or 1
    return {
        "clips": len(rows),
        "decisiones": dict(sorted(decisiones.items())),
        "drop_pct": round(drop / total * 100, 2),
        "keep_pct": round(keep / total * 100, 2),
        "retencion_keep_pct": round(keep / total * 100, 2),
        "retencion_keep_review_pct": round((keep + review) / total * 100, 2),
        "razones_principales": dict(razones.most_common(10)),
        "top_hard_fails": dict(hard_fails.most_common(10)),
        "top_review_reasons": dict(review_reasons.most_common(10)),
        "senales_no_aplicables_roi_mode": dict(non_applicable.most_common(10)),
        "senales_experimentales_no_decision": dict(experimental.most_common(10)),
        "sanity_checks": sanity_checks(rows),
        "por_split": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "input_kind": dict(sorted(Counter(row.get("input_kind", "") for row in rows).items())),
        "audit_confidence": dict(sorted(Counter(row.get("audit_confidence", "") for row in rows).items())),
        "fuentes": len({row.get("source_id", "") for row in rows}),
    }


def sanity_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_real = len(rows)
    total = total_real or 1
    decisiones = Counter(row.get("decision", "") for row in rows)
    drop_count = decisiones.get("drop", 0)
    keep_count = decisiones.get("keep", 0)
    drop_pct = drop_count / total * 100
    keep_pct = keep_count / total * 100
    warnings: list[str] = []

    conteos_por_scope = {
        "used_for_decision": contar_reasons(rows, "used_for_decision_reasons", fallback_columna="reasons"),
        "hard_fail": contar_reasons(rows, "hard_fail_reasons"),
        "review": contar_reasons(rows, "review_reasons"),
        "experimental": contar_reasons(rows, "experimental_reasons"),
        "not_applicable": contar_reasons(rows, "non_applicable_reasons"),
    }
    razones_no_discriminativas = []
    for scope, conteos in conteos_por_scope.items():
        for reason, count in conteos.items():
            pct = count / total * 100
            if pct > 90.0:
                razones_no_discriminativas.append(
                    {"razon": reason, "scope": scope, "clips": count, "pct": round(pct, 2)}
                )
    razones_no_discriminativas.sort(key=lambda item: (-item["pct"], item["razon"]))

    experimentales_masivas = []
    for reason, count in conteos_por_scope["experimental"].items():
        pct = count / total * 100
        if pct > 30.0:
            experimentales_masivas.append({"razon": reason, "clips": count, "pct": round(pct, 2)})
    experimentales_masivas.sort(key=lambda item: (-item["pct"], item["razon"]))

    if drop_pct > 5.0:
        warnings.append(
            f"ADVERTENCIA FUERTE: drop_pct={drop_pct:.2f}% supera 5%; inspeccionar ejemplos visualmente antes de filtrar."
        )
    if keep_pct < 60.0:
        warnings.append(
            f"ADVERTENCIA FUERTE: keep_pct={keep_pct:.2f}% queda debajo de 60%; filtro agresivo o senales dudosas."
        )
    for item in razones_no_discriminativas:
        warnings.append(
            f"Reason no discriminativa: {item['razon']} aparece en {item['pct']:.2f}% de clips ({item['scope']})."
        )
    for item in experimentales_masivas:
        warnings.append(
            f"Reason experimental masiva: {item['razon']} aparece en {item['pct']:.2f}% y no debe usarse para decidir."
        )

    drop_reason_dominante = detectar_drop_reason_dominante(rows, drop_count)
    if drop_reason_dominante:
        warnings.append(
            "ADVERTENCIA FUERTE: una sola razon explica casi todos los drops; "
            "revisar ejemplos antes de aceptar el filtro."
        )

    return {
        "drop_pct": round(drop_pct, 2),
        "keep_pct": round(keep_pct, 2),
        "razones_no_discriminativas": razones_no_discriminativas,
        "experimentales_masivas": experimentales_masivas,
        "drop_reason_dominante": drop_reason_dominante,
        "warnings": warnings,
    }


def detectar_drop_reason_dominante(rows: list[dict[str, Any]], drop_count: int) -> dict[str, Any]:
    if drop_count == 0:
        return {}
    drop_rows = [row for row in rows if row.get("decision") == "drop"]
    conteos = Counter(
        reason
        for row in drop_rows
        for reason in _split_reasons(row.get("hard_fail_reasons") or row.get("used_for_decision_reasons") or row.get("reasons"))
    )
    if not conteos:
        return {}
    reason, count = conteos.most_common(1)[0]
    pct_drops = count / max(drop_count, 1) * 100
    if pct_drops < 80.0:
        return {}
    ejemplos = [
        {
            "source_id": row.get("source_id", ""),
            "clip": row.get("clip", ""),
            "path_roi": row.get("path_roi", ""),
            "path_video": row.get("path_video", ""),
            "used_for_decision_reasons": row.get("used_for_decision_reasons") or row.get("reasons", ""),
        }
        for row in drop_rows
        if reason in _split_reasons(row.get("hard_fail_reasons") or row.get("used_for_decision_reasons") or row.get("reasons"))
    ][:5]
    return {"razon": reason, "clips": count, "pct_drops": round(pct_drops, 2), "ejemplos": ejemplos}


def contar_reasons(rows: list[dict[str, Any]], columna: str, *, fallback_columna: str | None = None) -> Counter[str]:
    return Counter(
        reason
        for row in rows
        for reason in _split_reasons(row.get(columna) or (row.get(fallback_columna) if fallback_columna else ""))
    )


def _split_reasons(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def leer_texto(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def repo_rel(path: str | Path | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    try:
        return path.resolve().relative_to(RAIZ_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def rostro_no_disponible(nota: str) -> dict[str, Any]:
    return {
        "face_detector": "no_disponible",
        "frames_rostro_muestreados": 0,
        "face_count_score": 1.0,
        "track_stability_score": 1.0,
        "pose_score": 0.5,
        "pose_available": False,
        "pose_reason": "not_available",
        "multi_face_risk": 0.0,
        "ratio_cara": "",
        "avg_caras": "",
        "max_caras": "",
        "dominant_face_shift": "",
        "face_box_area_jitter": "",
        "face_boxes_summary": "",
        "face_notes": nota,
    }


def _paths(input_path: Path | None, input_kind: str, path_roi: Path | None, path_video: Path, path_text: Path) -> dict[str, Any]:
    return {
        "input_path": input_path,
        "input_kind": input_kind,
        "path_roi": path_roi,
        "path_video": path_video,
        "path_text": path_text,
    }


def _repo_path_or_none(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else RAIZ_REPO / path


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "si"}


def _clip_number(clip_id: str) -> int:
    digits = "".join(ch for ch in clip_id if ch.isdigit())
    return int(digits) if digits else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria visual offline del dataset VSR.")
    parser.add_argument("--split", required=True, help="CSV de split con columnas titulo, clip, texto y opcional npz.")
    parser.add_argument("--source-id", help="Auditar una sola fuente/titulo.")
    parser.add_argument("--limit", type=int, help="Alias historico de --max-clips.")
    parser.add_argument("--max-clips", type=int, help="Cantidad maxima total de clips a auditar.")
    parser.add_argument("--sample-strategy", choices=["first", "stratified"], default="first")
    parser.add_argument("--clips-per-source", type=int, help="Cantidad de clips por fuente para muestreo estratificado.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Manifest CSV de salida.")
    parser.add_argument("--keep-output", help="Manifest candidato filtrado solo con decision=keep.")
    parser.add_argument("--keep-review-output", help="Manifest candidato con decision=keep o review.")
    parser.add_argument("--require-roi", action="store_true", help="Falla si la cobertura ROI queda debajo de --min-roi-coverage.")
    parser.add_argument("--min-roi-coverage", type=float, default=0.8, help="Cobertura minima de ROIs para considerar interpretable la corrida.")
    parser.add_argument("--allow-raw-fallback", action="store_true", help="Permite explicitamente auditar con fallback raw_clip.")
    parser.add_argument("--preflight-only", action="store_true", help="Solo revisa disponibilidad de ROIs y paths, sin auditar.")
    parser.add_argument("--preview-dir", help="Directorio opcional para hojas de contacto JPG.")
    parser.add_argument("--max-frames", type=int, default=64, help="Frames muestreados por clip para metricas visuales.")
    parser.add_argument("--face-frames", type=int, default=4, help="Frames muestreados para detector liviano de caras.")
    parser.add_argument("--sin-haar", action="store_true", help="Desactiva el detector Haar de caras.")
    parser.add_argument("--min-mouth-activity-review", type=float, default=Thresholds.min_mouth_activity_review)
    parser.add_argument("--min-mouth-visibility-review", type=float, default=Thresholds.min_mouth_visibility_review)
    parser.add_argument("--max-scene-cut-review", type=float, default=Thresholds.max_scene_cut_review)
    args = parser.parse_args()

    thresholds = Thresholds(
        min_mouth_activity_review=args.min_mouth_activity_review,
        min_mouth_visibility_review=args.min_mouth_visibility_review,
        max_scene_cut_review=args.max_scene_cut_review,
    )
    preflight_rows = cargar_split(
        args.split,
        source_id=args.source_id,
        limit=args.limit,
        max_clips=args.max_clips,
        sample_strategy=args.sample_strategy,
        clips_per_source=args.clips_per_source,
        seed=args.seed,
    )
    preflight = preflight_rois(preflight_rows, min_roi_coverage=args.min_roi_coverage)
    imprimir_preflight(preflight, allow_raw_fallback=args.allow_raw_fallback)
    try:
        validar_preflight(preflight, require_roi=args.require_roi)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    if args.preflight_only:
        return

    rows = auditar_split(
        split_path=args.split,
        output_path=args.output,
        source_id=args.source_id,
        limit=args.limit,
        max_clips=args.max_clips,
        sample_strategy=args.sample_strategy,
        clips_per_source=args.clips_per_source,
        seed=args.seed,
        max_frames=args.max_frames,
        face_frames=args.face_frames,
        usar_haar=not args.sin_haar,
        thresholds=thresholds,
        min_roi_coverage=args.min_roi_coverage,
        require_roi=args.require_roi,
    )

    if args.preview_dir:
        from data_cleaning.src.visual_quality_report import generar_previews_manifest

        generar_previews_manifest(rows, args.preview_dir)
    if args.keep_output:
        from data_cleaning.src.visual_quality_report import escribir_manifest_filtrado

        escribir_manifest_filtrado(rows, args.keep_output, decisiones={"keep"})
    if args.keep_review_output:
        from data_cleaning.src.visual_quality_report import escribir_manifest_filtrado

        escribir_manifest_filtrado(rows, args.keep_review_output, decisiones={"keep", "review"})

    print(json.dumps(resumen(rows), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Manifest: {repo_rel(args.output)}")
    if args.keep_output:
        print(f"Manifest keep: {repo_rel(args.keep_output)}")
    if args.keep_review_output:
        print(f"Manifest keep+review: {repo_rel(args.keep_review_output)}")


if __name__ == "__main__":
    main()
