"""Tablas y vistas para revisar lower_face_resized96."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"


def cargar_smoke(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "preprocessing_variant_manifest_smoke.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def cargar_full(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    path = output_base / "preprocessing_variant_manifest_full.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def smoke_status(smoke: pd.DataFrame) -> str:
    if smoke.empty:
        return "ready"
    statuses = set(smoke["status"].astype(str))
    reasons = set(smoke.get("reason", pd.Series(dtype=str)).astype(str))
    if "ok" in statuses:
        return "generated"
    if "blocked" in statuses and any("blocked_missing_dependency_mediapipe" in r for r in reasons):
        return "blocked_missing_dependency_mediapipe"
    return ",".join(sorted(statuses))


def smoke_decision(smoke: pd.DataFrame) -> str:
    if smoke.empty:
        return "NOT_READY_FOR_TRAINING"
    statuses = set(smoke["status"].astype(str))
    if statuses == {"ok"}:
        return "READY_FOR_FULL_GENERATION"
    if "blocked" in statuses:
        return "NOT_READY_FOR_TRAINING"
    return "BLOCKED_REVIEW_NEEDED"


def _npz_info(path: str | Path) -> tuple[str, str, int]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return "", "", 0
    arr = np.load(p)["rois"]
    return "x".join(str(x) for x in arr.shape), str(arr.dtype), int(arr.shape[0])


def sample_table(smoke: pd.DataFrame) -> pd.DataFrame:
    if smoke.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "clip",
                "current_shape",
                "variant_shape",
                "dtype",
                "frame_count",
                "status",
                "reason",
            ]
        )
    rows = []
    for _, row in smoke.iterrows():
        current_shape, current_dtype, current_frames = _npz_info(row["original_roi_path"])
        variant_shape, variant_dtype, variant_frames = _npz_info(row["variant_roi_path"])
        rows.append(
            {
                "source_id": row["source_id"],
                "clip": row["clip"],
                "current_shape": current_shape,
                "variant_shape": variant_shape,
                "dtype": variant_dtype or current_dtype,
                "frame_count": variant_frames or current_frames,
                "status": row["status"],
                "reason": row.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def preview_paths(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    preview_dir = output_base / "preprocessing_variant_preview"
    rows = [{"path": str(path)} for path in sorted(preview_dir.glob("*.png"))]
    return pd.DataFrame(rows)


def preview_groups(output_base: Path = OUTPUT_BASE) -> pd.DataFrame:
    preview_dir = output_base / "preprocessing_variant_preview"
    grouped: dict[str, dict[str, str]] = {}
    suffixes = {
        "__current_grid.png": "current_roi_96x96",
        "__lower_face_resized96_grid.png": "lower_face_resized96",
        "__side_by_side.png": "side_by_side",
    }
    for path in sorted(preview_dir.glob("*.png")):
        name = path.name
        for suffix, column in suffixes.items():
            if name.endswith(suffix):
                key = name[: -len(suffix)]
                grouped.setdefault(key, {})[column] = str(path)
                break
    rows = []
    for key, values in sorted(grouped.items()):
        rows.append({"sample": key, **values})
    return pd.DataFrame(rows)


def pending_message(status: str, decision: str | None = None) -> str:
    if status == "generated" and decision == "READY_FOR_FULL_GENERATION":
        return "Smoke generado: revisar grillas y shapes antes de VM full."
    if status == "generated":
        return "Smoke generado con advertencias: revisar errores antes de VM full."
    return "Pendiente de VM: falta generar lower_face_resized96."


def full_generation_status(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return pd.DataFrame([{"full_rows": 0, "ok": 0, "failed": 0, "blocked": 0, "status": "pending_generation"}])
    counts = full["status"].astype(str).value_counts()
    ok = int(counts.get("ok", 0))
    failed = int(counts.get("failed", 0))
    blocked = int(counts.get("blocked", 0))
    status = "ready" if ok == len(full) and failed == 0 and blocked == 0 else "review_needed"
    return pd.DataFrame([{"full_rows": len(full), "ok": ok, "failed": failed, "blocked": blocked, "status": status}])
