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


def pending_message(status: str) -> str:
    if status == "generated":
        return "Smoke generado: revisar grillas y shapes antes de VM full."
    return "Pendiente de VM: falta generar lower_face_resized96."
