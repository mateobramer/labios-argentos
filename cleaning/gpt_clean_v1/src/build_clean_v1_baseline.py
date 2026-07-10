"""Construye manifests clean_v1 conservadores.

No llama a modelos ni usa credenciales. El baseline conserva `large_existing`
como `clean_text` y marca por fila que GPT cleaning no fue aplicado.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "cleaning/gpt_clean_v1"
RELEASE = REPO / "data_release"
OUT = ROOT / "outputs"
CONTEXT_PACKS = OUT / "context_packs"
LLM_RAW = OUT / "llm_outputs" / "raw"

ARG_EXISTING = RELEASE / "argentina_existing_manifest.csv"
ARG_NEW = RELEASE / "argentina_new_manifest.csv"

CLEAN_MANIFEST = OUT / "clean_manifest.csv"
PATCH_LOG = OUT / "patch_log_clean_v1.jsonl"
PATCH_CANDIDATES = OUT / "llm_patch_candidates.jsonl"
ASR_STATUS = RELEASE / "asr_status_manifest.csv"
DISAGREEMENT = RELEASE / "asr_disagreement.csv"
FAILURES = RELEASE / "reports" / "failures.csv"
CLEANING_REPORT = RELEASE / "reports" / "cleaning_report.md"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
SOURCE_LIP_ROIS = "gs://labios-argentos-vsr-dataset/lip_rois/"
DEST_EXISTING = f"{DEST_BUCKET}/argentina/existing"

CLEAN_FIELDS = [
    "dataset_group",
    "source_id",
    "clip_id",
    "split",
    "large_text",
    "turbo_text",
    "clean_text",
    "clean_status",
    "clean_confidence",
    "patch_count",
    "needs_review_reason",
    "mp4_path",
    "npz_path",
    "txt_large_path",
    "txt_turbo_path",
    "txt_clean_path",
    "url",
    "title",
    "channel",
    "notes",
]

ASR_FIELDS = [
    "dataset_group",
    "clip_count_mp4",
    "large_count_txt",
    "turbo_count_txt",
    "asr_status",
    "asr_model_large",
    "asr_model_turbo",
    "asr_runtime",
    "asr_error_if_any",
    "notes",
]

DISAGREEMENT_FIELDS = [
    "dataset_group",
    "clip_id",
    "cer",
    "wer",
    "length_ratio",
    "status",
    "notes",
]

FAILURE_FIELDS = ["stage", "dataset_group", "item", "status", "error", "notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def source_id(row: dict[str, str]) -> str:
    raw = f"{row.get('spk', '')}__{row.get('titulo', '')}".strip("_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return value[:120] or "unknown_source"


def dest_path(source_path: str, kind: str) -> str:
    if not source_path.startswith(SOURCE_LIP_ROIS):
        return ""
    suffix = source_path[len(SOURCE_LIP_ROIS) :]
    if kind == "mp4":
        return f"{DEST_EXISTING}/clips_mp4/{suffix}"
    if kind == "npz":
        return f"{DEST_EXISTING}/rois_npz/{suffix}"
    if kind == "large":
        return f"{DEST_EXISTING}/transcripts/large/{suffix}"
    if kind == "clean":
        return f"{DEST_EXISTING}/transcripts/clean_v1/{suffix}"
    return ""


def build_clean_rows(existing_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    clean_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in existing_rows:
        sid = source_id(row)
        large_text = row.get("text_large_existing", "")
        clip_id = row.get("clip_id", "")
        clean_row = {
            "dataset_group": "argentina/existing",
            "source_id": sid,
            "clip_id": clip_id,
            "split": row.get("split", ""),
            "large_text": large_text,
            "turbo_text": "",
            "clean_text": large_text,
            "clean_status": "unchanged_no_llm_baseline",
            "clean_confidence": "low",
            "patch_count": 0,
            "needs_review_reason": "llm_browser_not_run;turbo_blocked_no_audio_in_roi_mp4",
            "mp4_path": dest_path(row.get("mp4_gcs_path", ""), "mp4"),
            "npz_path": dest_path(row.get("npz_gcs_path", ""), "npz"),
            "txt_large_path": dest_path(row.get("txt_gcs_path", ""), "large"),
            "txt_turbo_path": "",
            "txt_clean_path": dest_path(row.get("txt_gcs_path", ""), "clean"),
            "url": row.get("url", ""),
            "title": row.get("titulo", ""),
            "channel": "",
            "notes": "clean_v1 mirrors large_existing until GPT patches are validated",
        }
        clean_rows.append(clean_row)
        grouped[sid].append(clean_row)

    CONTEXT_PACKS.mkdir(parents=True, exist_ok=True)
    for sid, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: str(r.get("clip_id", "")))
        path = CONTEXT_PACKS / f"{sid}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for index, item in enumerate(rows_sorted):
                prev_text = rows_sorted[index - 1]["large_text"] if index else ""
                next_text = rows_sorted[index + 1]["large_text"] if index + 1 < len(rows_sorted) else ""
                record = {
                    "source_id": sid,
                    "url": item["url"],
                    "title": item["title"],
                    "clip_id": item["clip_id"],
                    "large_text": item["large_text"],
                    "turbo_text": "",
                    "prev_large_text": prev_text,
                    "next_large_text": next_text,
                    "disagreement_status": "blocked_missing_turbo_no_audio",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return clean_rows


def write_patch_outputs(clean_rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LLM_RAW.mkdir(parents=True, exist_ok=True)
    PATCH_CANDIDATES.write_text("", encoding="utf-8")
    with PATCH_LOG.open("w", encoding="utf-8", newline="\n") as f:
        for row in clean_rows:
            record = {
                "dataset_group": row["dataset_group"],
                "source_id": row["source_id"],
                "clip_id": row["clip_id"],
                "large_text": row["large_text"],
                "turbo_text": row["turbo_text"],
                "clean_text": row["clean_text"],
                "status": row["clean_status"],
                "confidence": row["clean_confidence"],
                "patches": [],
                "blocked_reasons": [
                    "turbo_blocked_no_audio_in_roi_mp4",
                    "llm_browser_not_run",
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_asr_status(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> None:
    asr_rows = [
        {
            "dataset_group": "argentina/existing",
            "clip_count_mp4": 12112,
            "large_count_txt": 12112,
            "turbo_count_txt": 0,
            "asr_status": "blocked_no_audio_in_roi_mp4",
            "asr_model_large": "existing_txt_from_source_bucket",
            "asr_model_turbo": "not_run",
            "asr_runtime": "",
            "asr_error_if_any": "sample probes showed video-only ROI mp4; audio stream absent",
            "notes": "Need original clips with audio or source reconstruction before turbo.",
        },
        {
            "dataset_group": "spanish_general/existing",
            "clip_count_mp4": 10356,
            "large_count_txt": 46991,
            "turbo_count_txt": 0,
            "asr_status": "blocked_no_audio_in_roi_mp4",
            "asr_model_large": "existing_txt_from_source_bucket",
            "asr_model_turbo": "not_run",
            "asr_runtime": "",
            "asr_error_if_any": "sample probes showed video-only ROI mp4; audio stream absent",
            "notes": "Kept separated; no GPT cleaning in this phase.",
        },
        {
            "dataset_group": "argentina/new_discovery",
            "clip_count_mp4": 0,
            "large_count_txt": 0,
            "turbo_count_txt": 0,
            "asr_status": "pending_source_ingest",
            "asr_model_large": "large",
            "asr_model_turbo": "turbo",
            "asr_runtime": "",
            "asr_error_if_any": "",
            "notes": f"{len(new_rows)} accepted videos queued from data_discovery.",
        },
    ]
    write_csv(ASR_STATUS, asr_rows, ASR_FIELDS)
    write_csv(
        DISAGREEMENT,
        [
            {
                "dataset_group": "argentina/existing",
                "clip_id": "",
                "cer": "",
                "wer": "",
                "length_ratio": "",
                "status": "blocked_missing_turbo_no_audio",
                "notes": "No turbo transcript generated because ROI mp4 has no audio.",
            },
            {
                "dataset_group": "spanish_general/existing",
                "clip_id": "",
                "cer": "",
                "wer": "",
                "length_ratio": "",
                "status": "blocked_missing_turbo_no_audio",
                "notes": "No turbo transcript generated because ROI mp4 has no audio.",
            },
        ],
        DISAGREEMENT_FIELDS,
    )
    write_csv(
        FAILURES,
        [
            {
                "stage": "asr_turbo",
                "dataset_group": "argentina/existing",
                "item": "existing_roi_mp4",
                "status": "blocked_no_audio_in_roi_mp4",
                "error": "ffmpeg probe found video stream only; audio:0KiB",
                "notes": "Reconstruct from original URLs/raw videos to run turbo.",
            },
            {
                "stage": "asr_turbo",
                "dataset_group": "spanish_general/existing",
                "item": "existing_roi_mp4",
                "status": "blocked_no_audio_in_roi_mp4",
                "error": "ffmpeg probe found video stream only; audio:0KiB",
                "notes": "Kept separated; no GPT cleaning requested.",
            },
            {
                "stage": "new_discovery_ingest",
                "dataset_group": "argentina/new_discovery",
                "item": "accepted_video_queue",
                "status": "pending_source_ingest",
                "error": "",
                "notes": f"{len(new_rows)} accepted videos queued for source download and full ASR.",
            },
        ],
        FAILURE_FIELDS,
    )


def write_report(clean_rows: list[dict[str, object]], existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> None:
    status_counts = Counter(str(row["clean_status"]) for row in clean_rows)
    source_count = len({row["source_id"] for row in clean_rows})
    accepted_new = [row for row in new_rows if row.get("decision") in {"strong_accept", "accept"}]
    CLEANING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT.write_text(
        "\n".join(
            [
                "# Cleaning report clean_v1",
                "",
                f"argentina_existing_manifest_rows: {len(existing_rows)}",
                f"clean_manifest_rows: {len(clean_rows)}",
                f"context_pack_sources: {source_count}",
                f"status_counts: {dict(status_counts)}",
                f"argentina_new_accepted_videos_queued: {len(accepted_new)}",
                "",
                "## ASR status",
                "",
                "- Existing argentina ROI mp4 probe: video-only 96x96, audio stream absent.",
                "- Existing spanish_general ROI mp4 probe: video-only 96x96, audio stream absent.",
                "- Turbo ASR for existing data is blocked until clips with audio are reconstructed.",
                "",
                "## clean_v1 status",
                "",
                "`clean_v1` mirrors `large_existing` as a conservative baseline.",
                "No GPT patch was applied in this run, and every row is marked",
                "`unchanged_no_llm_baseline` with low confidence plus review reason.",
                "",
                "## Next scaling step",
                "",
                "Reconstruct audio-bearing clips from mapped URLs/raw sources, run large/turbo,",
                "then feed the generated context packs to the JSONL prompt and validate patches",
                "before replacing clean_v1 text.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    existing_rows = read_csv(ARG_EXISTING)
    new_rows = read_csv(ARG_NEW)
    clean_rows = build_clean_rows(existing_rows)
    write_csv(CLEAN_MANIFEST, clean_rows, CLEAN_FIELDS)
    write_patch_outputs(clean_rows)
    write_asr_status(existing_rows, new_rows)
    write_report(clean_rows, existing_rows, new_rows)
    print(f"clean_manifest_rows={len(clean_rows)} -> {CLEAN_MANIFEST}")
    print(f"patch_log -> {PATCH_LOG}")
    print(f"context_packs -> {CONTEXT_PACKS}")
    print(f"asr_status -> {ASR_STATUS}")
    print(f"cleaning_report -> {CLEANING_REPORT}")


if __name__ == "__main__":
    main()
