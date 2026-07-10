"""Smoke/full generation para la variante lower_face_resized96."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr_models" / "splits" / "splits.csv"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
VARIANT = "lower_face_resized96"
_WORKER_LANDMARKER = None
_WORKER_VPROC = None
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


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return safe.strip("_")[:120]


def _grid(frames: np.ndarray, n: int = 6) -> np.ndarray:
    if frames.ndim != 3 or frames.shape[0] == 0:
        return np.zeros((96, 96), dtype=np.uint8)
    idxs = np.linspace(0, frames.shape[0] - 1, min(n, frames.shape[0]), dtype=int)
    return np.concatenate([frames[i] for i in idxs], axis=1)


def _write_previews(row: dict[str, str], variant_path: Path, preview_dir: Path) -> dict[str, str]:
    original_roi = REPO_ROOT / row.get("npz", "")
    if not original_roi.exists() or not variant_path.exists():
        return {}
    current = np.load(original_roi)["rois"]
    variant = np.load(variant_path)["rois"]
    current_grid = _grid(current)
    variant_grid = _grid(variant)
    h = max(current_grid.shape[0], variant_grid.shape[0])
    if current_grid.shape[0] != h:
        current_grid = cv2.resize(current_grid, (current_grid.shape[1], h), interpolation=cv2.INTER_AREA)
    if variant_grid.shape[0] != h:
        variant_grid = cv2.resize(variant_grid, (variant_grid.shape[1], h), interpolation=cv2.INTER_AREA)
    side = np.concatenate([current_grid, variant_grid], axis=0)
    preview_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_name(f"{row['titulo']}__{row['clip']}")
    current_path = preview_dir / f"{prefix}__current_grid.png"
    variant_out = preview_dir / f"{prefix}__lower_face_resized96_grid.png"
    side_path = preview_dir / f"{prefix}__side_by_side.png"
    cv2.imwrite(str(current_path), current_grid)
    cv2.imwrite(str(variant_out), variant_grid)
    cv2.imwrite(str(side_path), side)
    return {
        "current_preview": str(current_path),
        "variant_preview": str(variant_out),
        "side_by_side_preview": str(side_path),
    }


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


def _fallback_original_roi(row: dict[str, str], variant_path: Path, original_reason: str) -> dict[str, str] | None:
    original_roi = REPO_ROOT / row.get("npz", "")
    if not original_roi.exists():
        return None

    try:
        with np.load(original_roi) as data:
            rois = data["rois"]
            shape = "x".join(str(x) for x in rois.shape)
            dtype = str(rois.dtype)
    except Exception:
        return None

    variant_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original_roi, variant_path)
    return {
        "source_id": row["titulo"],
        "clip": row["clip"],
        "original_roi_path": row.get("npz", ""),
        "variant_roi_path": str(variant_path),
        "variant": VARIANT,
        "shape": shape,
        "dtype": dtype,
        "status": "ok",
        "reason": f"fallback_original_roi_after_variant_no_frames; original_reason={original_reason}",
    }


def _generar_clip(
    row: dict[str, str],
    output_dir: Path,
    landmarker,
    vproc,
    preview_dir: Path | None = None,
    make_preview: bool = False,
) -> dict[str, str]:
    from preprocessing.src.preprocesar import (
        procesar_clip,
        remuestrear_a_25fps,
    )

    clip_path = REPO_ROOT / "data" / "clips" / row["titulo"] / f"{row['clip']}.mp4"
    variant_path = output_dir / row["titulo"] / f"{row['clip']}.npz"
    frames, ratio = procesar_clip(str(clip_path), landmarker, vproc)
    if not frames:
        reason = f"sin frames variant; detection_ratio={ratio:.3f}"
        fallback = _fallback_original_roi(row, variant_path, reason)
        if fallback is not None:
            return fallback
        return {
            "source_id": row["titulo"],
            "clip": row["clip"],
            "original_roi_path": row.get("npz", ""),
            "variant_roi_path": str(variant_path),
            "variant": VARIANT,
            "shape": "",
            "dtype": "",
            "status": "failed",
            "reason": reason,
        }

    resized = [cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA) for frame in frames]
    arr = np.asarray(remuestrear_a_25fps(resized, 25), dtype=np.uint8)
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(variant_path, rois=arr)
    result = {
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
    if make_preview and preview_dir is not None:
        _write_previews(row, variant_path, preview_dir)
    return result


def _init_worker() -> None:
    global _WORKER_LANDMARKER, _WORKER_VPROC

    from preprocessing.src.preprocesar import crear_landmarker
    from preprocessing.src.video_process import VideoProcess

    _WORKER_LANDMARKER = crear_landmarker()
    # Crop mas amplio alrededor de la boca usando el pipeline existente; resize final a 96x96.
    _WORKER_VPROC = VideoProcess(crop_width=128, crop_height=128, convert_gray=True)


def _generar_clip_worker(args: tuple[dict[str, str], str, str, bool]) -> dict[str, str]:
    row, output_dir, preview_dir, make_preview = args
    if _WORKER_LANDMARKER is None or _WORKER_VPROC is None:
        _init_worker()
    out = Path(output_dir)
    try:
        return _generar_clip(
            row,
            out,
            _WORKER_LANDMARKER,
            _WORKER_VPROC,
            Path(preview_dir),
            make_preview,
        )
    except Exception as exc:  # pragma: no cover - depende de videos/mediapipe local
        return {
            "source_id": row["titulo"],
            "clip": row["clip"],
            "original_roi_path": row.get("npz", ""),
            "variant_roi_path": str(out / row["titulo"] / f"{row['clip']}.npz"),
            "variant": VARIANT,
            "shape": "",
            "dtype": "",
            "status": "failed",
            "reason": str(exc),
        }


def run_preprocessing_variant(
    splits_path: Path = DEFAULT_SPLITS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    max_clips: int = 2,
    full: bool = False,
    preview_max: int = 20,
    workers: int = 1,
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
    preview_dir = output_base / "preprocessing_variant_preview"

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
    previews = min(preview_max, len(selected))
    if workers > 1 and len(selected) > 1:
        tasks = [
            (row, str(output_dir), str(preview_dir), idx < preview_max)
            for idx, row in enumerate(selected)
        ]
        context = mp.get_context("spawn")
        with context.Pool(processes=workers, initializer=_init_worker) as pool:
            manifest_rows = pool.map(_generar_clip_worker, tasks)
    else:
        _init_worker()
        for idx, row in enumerate(selected):
            try:
                generated = _generar_clip(
                    row,
                    output_dir,
                    _WORKER_LANDMARKER,
                    _WORKER_VPROC,
                    preview_dir,
                    idx < preview_max,
                )
                manifest_rows.append(generated)
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
        "preview_dir": str(preview_dir),
        "previews": sum(
            1
            for row in manifest_rows[:previews]
            if row["status"] == "ok"
        ),
        "workers": workers,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    ap.add_argument("--max-clips", type=int, default=2)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--preview-max", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preprocessing_variant(
        splits_path=args.splits,
        output_base=args.output_base,
        max_clips=args.max_clips,
        full=args.full,
        preview_max=args.preview_max,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
