"""Genera contact sheets livianos desde samples descargados."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_pipeline.discovery.src.common import (
    CONTACT_SHEETS,
    SAMPLE_METADATA,
    SAMPLES,
    asegurar_directorios,
    cargar_json,
    configurar_salida_utf8,
    to_float,
)


def leer_frames(path: Path, frames_por_sample: int = 4, ancho: int = 220) -> list[Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        indices = [int((i + 0.5) * total / frames_por_sample) for i in range(frames_por_sample)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            escala = ancho / max(w, 1)
            frame = cv2.resize(frame, (ancho, max(1, int(h * escala))), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def agregar_label(frame: Any, text: str) -> Any:
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    label = np.full((42, w, 3), 245, dtype=np.uint8)
    cv2.putText(label, text[:70], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([label, frame])


def armar_grid(frames: list[Any], columnas: int = 4) -> Any:
    import cv2
    import numpy as np

    if not frames:
        return None
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for frame in frames:
        h, w = frame.shape[:2]
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        canvas[:h, :w] = frame
        padded.append(canvas)
    rows = []
    for i in range(0, len(padded), columnas):
        row_frames = padded[i : i + columnas]
        while len(row_frames) < columnas:
            row_frames.append(np.full((max_h, max_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(row_frames))
    return np.vstack(rows)


def generar_contact_sheet(audit_path: Path, output_dir: Path = CONTACT_SHEETS) -> Path | None:
    import cv2

    audit = cargar_json(audit_path)
    video_id = str(audit.get("video_id") or audit_path.stem)
    frames = []
    for sample in audit.get("samples", []):
        sample_path = Path(str(sample.get("sample_path") or ""))
        if not sample_path.exists():
            fallback_dir = SAMPLES / video_id
            candidates = sorted(fallback_dir.glob(f"sample_{int(sample.get('index', 0)):02d}.*"))
            sample_path = candidates[0] if candidates else sample_path
        if not sample_path.exists():
            continue
        label = f"{video_id} s{sample.get('index')} {sample.get('start')}-{sample.get('end')}s V={sample.get('visual_quality_score', '')}"
        for frame in leer_frames(sample_path):
            frames.append(agregar_label(frame, label))
    if not frames:
        return None
    title = agregar_titulo(
        armar_grid(frames),
        f"{video_id} | {audit.get('channel', '')} | visual={audit.get('visual_quality_score', '')} | decision={audit.get('visual_decision', '')}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{video_id}.jpg"
    ok = cv2.imwrite(str(out), cv2.cvtColor(title, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return out if ok else None


def agregar_titulo(img: Any, title: str) -> Any:
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    bar = np.full((55, w, 3), 238, dtype=np.uint8)
    cv2.putText(bar, title[:130], (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    configurar_salida_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-metadata", type=Path, default=SAMPLE_METADATA)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--accepted-only", action="store_true")
    args = parser.parse_args()

    asegurar_directorios()
    count = 0
    for audit_path in sorted(args.sample_metadata.glob("*.json")):
        audit = cargar_json(audit_path)
        if args.accepted_only:
            if audit.get("visual_decision") not in {"strong_accept", "accept", "maybe_review"}:
                continue
            if to_float(audit.get("visual_quality_score")) <= 0:
                continue
        out = generar_contact_sheet(audit_path)
        if out:
            count += 1
            print(f"contact_sheet -> {out}")
        if count >= args.limit:
            break
    print(f"contact_sheets={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
