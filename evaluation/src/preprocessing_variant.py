"""Smoke/full generation para la variante lower_face_resized96."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr_models" / "splits" / "splits.csv"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
VARIANT = "lower_face_resized96"
MANIFEST_COLUMNS = [
    "source_id",
    "clip",
    "original_roi_path",
    "variant_roi_path",
    "variant",
    "shape",
    "dtype",
    "status",
    "reason",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _candidate_rows(rows: list[dict[str, str]], max_clips: int) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        clip_path = REPO_ROOT / "data" / "clips" / row["titulo"] / f"{row['clip']}.mp4"
        roi_path = REPO_ROOT / row.get("npz", "")
        if clip_path.exists() and roi_path.exists():
            candidates.append(row)
        if len(candidates) >= max_clips:
            break
    return candidates


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _blocked_rows(rows: list[dict[str, str]], output_dir: Path, reason: str) -> list[dict[str, str]]:
    blocked = []
    for row in rows:
        variant_path = output_dir / row["titulo"] / f"{row['clip']}.npz"
        blocked.append(
            {
                "source_id": row["titulo"],
                "clip": row["clip"],
                "original_roi_path": row.get("npz", ""),
                "variant_roi_path": str(variant_path),
                "variant": VARIANT,
                "shape": "",
                "dtype": "",
                "status": "blocked",
                "reason": reason,
            }
        )
    return blocked


def _generar_clip(row: dict[str, str], output_dir: Path) -> dict[str, str]:
    from visual_preprocessing.src.preprocesar import (
        crear_landmarker,
        procesar_clip,
        remuestrear_a_25fps,
    )
    from visual_preprocessing.src.video_process import VideoProcess

    clip_path = REPO_ROOT / "data" / "clips" / row["titulo"] / f"{row['clip']}.mp4"
    variant_path = output_dir / row["titulo"] / f"{row['clip']}.npz"
    landmarker = crear_landmarker()
    # Crop mas amplio alrededor de la boca usando el pipeline existente; resize final a 96x96.
    vproc = VideoProcess(crop_width=128, crop_height=128, convert_gray=True)
    frames, ratio = procesar_clip(str(clip_path), landmarker, vproc)
    if not frames:
        return {
            "source_id": row["titulo"],
            "clip": row["clip"],
            "original_roi_path": row.get("npz", ""),
            "variant_roi_path": str(variant_path),
            "variant": VARIANT,
            "shape": "",
            "dtype": "",
            "status": "failed",
            "reason": f"sin frames variant; detection_ratio={ratio:.3f}",
        }

    resized = [cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA) for frame in frames]
    arr = np.asarray(remuestrear_a_25fps(resized, 25), dtype=np.uint8)
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(variant_path, rois=arr)
    return {
        "source_id": row["titulo"],
        "clip": row["clip"],
        "original_roi_path": row.get("npz", ""),
        "variant_roi_path": str(variant_path),
        "variant": VARIANT,
        "shape": "x".join(str(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "status": "ok",
        "reason": "",
    }


def run_preprocessing_variant(
    splits_path: Path = DEFAULT_SPLITS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    max_clips: int = 2,
    full: bool = False,
) -> dict[str, object]:
    rows = leer_csv(splits_path)
    selected = rows if full else _candidate_rows(rows, max_clips)
    output_dir = (
        output_base / "rois_lower_face_resized96"
        if full
        else output_base / "preprocessing_variant_smoke"
    )
    manifest = (
        output_base / "preprocessing_variant_manifest_full.csv"
        if full
        else output_base / "preprocessing_variant_manifest_smoke.csv"
    )

    if importlib.util.find_spec("mediapipe") is None:
        reason = "blocked_missing_dependency_mediapipe"
        manifest_rows = _blocked_rows(selected, output_dir, reason)
        _write_manifest(manifest, manifest_rows)
        return {
            "variant": VARIANT,
            "mode": "full" if full else "smoke",
            "status": "blocked",
            "reason": reason,
            "clips": len(selected),
            "manifest": str(manifest),
            "output_dir": str(output_dir),
        }

    manifest_rows = []
    for row in selected:
        try:
            manifest_rows.append(_generar_clip(row, output_dir))
        except Exception as exc:  # pragma: no cover - depende de videos/mediapipe local
            manifest_rows.append(
                {
                    "source_id": row["titulo"],
                    "clip": row["clip"],
                    "original_roi_path": row.get("npz", ""),
                    "variant_roi_path": str(output_dir / row["titulo"] / f"{row['clip']}.npz"),
                    "variant": VARIANT,
                    "shape": "",
                    "dtype": "",
                    "status": "failed",
                    "reason": str(exc),
                }
            )
    _write_manifest(manifest, manifest_rows)
    ok = sum(1 for row in manifest_rows if row["status"] == "ok")
    return {
        "variant": VARIANT,
        "mode": "full" if full else "smoke",
        "status": "ok" if ok == len(manifest_rows) else "partial",
        "clips": len(manifest_rows),
        "ok": ok,
        "manifest": str(manifest),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    ap.add_argument("--max-clips", type=int, default=2)
    ap.add_argument("--full", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preprocessing_variant(
        splits_path=args.splits,
        output_base=args.output_base,
        max_clips=args.max_clips,
        full=args.full,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
