"""Estimacion conservadora de minutos utiles y clips aceptados."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_pipeline.discovery.src.common import clamp, leer_csv, to_float


DEFAULT_CLIPS_PER_MINUTE = 12.0


def estimar_minutos_utiles(
    *,
    video_duration_minutes: float,
    speech_presence_ratio: float,
    mouth_visible_ratio: float,
    single_speaker_ratio: float,
    visual_accept_ratio: float,
) -> float:
    return max(
        0.0,
        video_duration_minutes
        * clamp(speech_presence_ratio)
        * clamp(mouth_visible_ratio)
        * clamp(single_speaker_ratio)
        * clamp(visual_accept_ratio),
    )


def estimar_clips_aceptados(usable_minutes: float, clips_per_minute: float = DEFAULT_CLIPS_PER_MINUTE) -> float:
    return max(0.0, usable_minutes * clips_per_minute)


def estimar_desde_row(row: dict[str, Any], clips_per_minute: float = DEFAULT_CLIPS_PER_MINUTE) -> dict[str, float]:
    duration = to_float(row.get("duration_minutes"))
    speech = to_float(row.get("speech_presence_ratio"), 0.65)
    mouth = to_float(row.get("mouth_visible_ratio"), to_float(row.get("visual_quality_score"), 70.0) / 100.0)
    single = to_float(row.get("single_speaker_visual_proxy"), 0.75)
    visual = to_float(row.get("visual_quality_score"), 70.0) / 100.0
    usable = estimar_minutos_utiles(
        video_duration_minutes=duration,
        speech_presence_ratio=speech,
        mouth_visible_ratio=mouth,
        single_speaker_ratio=single,
        visual_accept_ratio=visual,
    )
    return {
        "usable_minutes_estimate": round(usable, 3),
        "accepted_clips_estimate": round(estimar_clips_aceptados(usable, clips_per_minute), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV con columnas de duracion y ratios.")
    parser.add_argument("--clips-per-minute", type=float, default=DEFAULT_CLIPS_PER_MINUTE)
    args = parser.parse_args()
    rows = leer_csv(args.csv)
    total_min = 0.0
    total_clips = 0.0
    for row in rows:
        est = estimar_desde_row(row, args.clips_per_minute)
        total_min += est["usable_minutes_estimate"]
        total_clips += est["accepted_clips_estimate"]
    print(f"usable_minutes_estimate={total_min:.1f}")
    print(f"accepted_clips_estimate={total_clips:.0f}")
    print(f"clips_per_minute_estimate={args.clips_per_minute:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

