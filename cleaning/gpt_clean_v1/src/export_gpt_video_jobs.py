"""Exporta jobs de limpieza GPT por video completo."""

from __future__ import annotations

import csv
import json
import re
import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "cleaning/gpt_clean_v1"
RELEASE = REPO / "data_release"
OUT = ROOT / "video_jobs"
PROMPT_TEMPLATE = ROOT / "prompts" / "transcript_clean_v1_prompt.md"

FINAL_RELEASE = RELEASE / "final_release_manifest.csv"
CLEAN_GPT = RELEASE / "clean_gpt_manifest.csv"
DISAGREEMENT = RELEASE / "asr_disagreement_v2.csv"
JOB_INDEX = OUT / "video_job_index.csv"
JOB_QUEUE = OUT / "video_job_queue.csv"

INDEX_FIELDS = [
    "job_id",
    "video_id",
    "safe_video_id",
    "part_index",
    "part_count",
    "job_kind",
    "dataset_group",
    "source_id",
    "title",
    "channel",
    "source_url",
    "clips_eligible",
    "main_clips",
    "context_clips",
    "estimated_chars",
    "prompt_path",
    "input_jsonl_path",
    "raw_output_path",
    "status",
    "created_at",
    "notes",
]

QUEUE_FIELDS = [
    "job_id",
    "video_id",
    "safe_video_id",
    "part_index",
    "part_count",
    "job_kind",
    "status",
    "clips_eligible",
    "main_clips",
    "context_clips",
    "estimated_chars",
    "prompt_path",
    "input_jsonl_path",
    "raw_output_path",
    "validated_path",
    "applied_at",
    "chat_url",
    "error",
    "updated_at",
]

FAILURE_FIELDS = [
    "video_id",
    "clip_id",
    "reason",
    "estimated_chars",
    "notes",
]

SPLIT_REPORT = OUT / "video_job_split_report.md"
CONTEXT_FAILURES = OUT / "video_job_context_failures.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe[:120] or "unknown_video"


def clip_sort_key(row: dict[str, str]) -> tuple[float, str]:
    try:
        start = float(row.get("start_time") or 0.0)
    except ValueError:
        start = 0.0
    return start, row.get("clip_id", "")


def completed_gpt_clip_ids() -> set[str]:
    return {
        row.get("clip_id", "")
        for row in read_csv(CLEAN_GPT)
        if row.get("status") in {"completed_clean_gpt", "rejected_clean_gpt"}
        or row.get("gpt_status") in {"completed_clean_gpt", "rejected_clean_gpt"}
    }


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main_clip_ids_from_input(path: Path) -> set[str]:
    return {
        row.get("clip_id", "")
        for row in read_jsonl(path)
        if row.get("clip_id") and not truthy(row.get("context_only"))
    }


def disagreement_by_clip() -> dict[str, dict[str, str]]:
    return {row.get("clip_id", ""): row for row in read_csv(DISAGREEMENT)}


def eligible_rows() -> list[dict[str, str]]:
    done = completed_gpt_clip_ids()
    rows = []
    for row in read_csv(FINAL_RELEASE):
        if row.get("dataset_group") not in {"argentina/new_discovery", "argentina/existing"}:
            continue
        if row.get("clip_id", "") in done:
            continue
        if not row.get("large_text", "").strip() or not row.get("turbo_text", "").strip():
            continue
        rows.append(row)
    return rows


def video_id_for(row: dict[str, str]) -> str:
    return row.get("source_video_id", "").strip() or row.get("source_id", "").strip() or "unknown_video"


def build_record(
    row: dict[str, str],
    disagreement: dict[str, dict[str, str]],
    job_id: str,
    context_only: bool = False,
) -> dict[str, object]:
    clip_id = row.get("clip_id", "")
    dis = disagreement.get(clip_id, {})
    flags = []
    if row.get("large_text", "").strip() != row.get("turbo_text", "").strip():
        flags.append("large_turbo_disagree")
    try:
        if float(dis.get("wer") or 0.0) >= 0.20:
            flags.append("high_wer_disagreement")
    except ValueError:
        pass
    return {
        "video_id": video_id_for(row),
        "job_id": job_id,
        "context_only": context_only,
        "dataset_group": row.get("dataset_group", ""),
        "source_id": row.get("source_id", ""),
        "clip_id": clip_id,
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "baseline_text": row.get("existing_text", ""),
        "large_text": row.get("large_text", ""),
        "turbo_text": row.get("turbo_text", ""),
        "selected_asr_text": row.get("large_text", "") or row.get("turbo_text", ""),
        "disagreement_flags": flags,
        "disagreement_wer": dis.get("wer", ""),
        "disagreement_cer": dis.get("cer", ""),
        "roi_npz_path": row.get("npz_path", ""),
        "usable_for_training": row.get("usable_for_training", ""),
    }


def render_prompt(video: dict[str, str], input_lines: list[str]) -> str:
    base_prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8") if PROMPT_TEMPLATE.exists() else ""
    schema = {
        "clip_id": "...",
        "action": "keep | patch | reject",
        "clean_text": "...",
        "reason": "...",
        "confidence": "high | medium | low",
        "notes": "...",
    }
    return "\n".join(
        [
            "# GPT cleaning por video completo",
            "",
            "Unidad de trabajo: un video_id completo. Devolve solo JSONL estricto, una linea por clip elegible.",
            "",
            "## Contexto del video",
            "",
            f"- video_id: {video.get('video_id', '')}",
            f"- dataset_group: {video.get('dataset_group', '')}",
            f"- source_id: {video.get('source_id', '')}",
            f"- title: {video.get('title', '')}",
            f"- channel: {video.get('channel', '')}",
            f"- source_url: {video.get('source_url', '')}",
            "",
            "## Reglas conservadoras",
            "",
            base_prompt.strip(),
            "",
            "- ROI no es requisito para limpiar texto.",
            "- Los records con `context_only=true` son solo continuidad de contexto; no devuelvas ninguna salida para esos clip_id.",
            "- Devolve salida solo para clips principales (`context_only=false`).",
            "- No inventar frases ni completar huecos.",
            "- No transformar a espanol idealizado ni borrar disfluencias reales.",
            "- Si large y turbo discrepan mucho, elegir la opcion conservadora o `reject`.",
            "- Si no esta claro, usar `keep` o `reject`.",
            "- Mantener registro argentino/informal/voseo.",
            "- No agregar puntuacion excesiva si cambia sentido.",
            "",
            "## Schema de salida",
            "",
            "Una linea JSON por clip, sin Markdown:",
            "",
            json.dumps(schema, ensure_ascii=False),
            "",
            "Acciones:",
            "- `keep`: conservar el ASR seleccionado; `clean_text` debe ser el texto final conservado.",
            "- `patch`: aplicar una correccion puntual con evidencia; `clean_text` no puede estar vacio.",
            "- `reject`: no aplicar texto limpio para ese clip; no se usara patch.",
            "",
            "## Input JSONL",
            "",
            *input_lines,
            "",
        ]
    )


def prompt_for_records(video: dict[str, str], records: list[dict[str, object]]) -> tuple[str, list[str]]:
    input_lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return render_prompt(video, input_lines), input_lines


def job_name(video_id: str, part_index: int | None = None) -> str:
    base = f"video_{safe_name(video_id)}"
    if part_index is None:
        return base
    return f"{base}__part_{part_index:03d}"


def records_for_part_window(
    rows: list[dict[str, str]],
    start: int,
    end: int,
    video_meta: dict[str, str],
    disagreement: dict[str, dict[str, str]],
    part_index: int | None,
    overlap_clips: int,
) -> list[dict[str, object]]:
    jid = job_name(video_meta["video_id"], part_index)
    overlap = max(0, overlap_clips)
    before_start = max(0, start - overlap)
    after_end = min(len(rows), end + overlap)
    records = []
    for idx in range(before_start, after_end):
        context_only = idx < start or idx >= end
        records.append(build_record(rows[idx], disagreement, jid, context_only=context_only))
    return records


def range_prompt_chars(
    rows: list[dict[str, str]],
    start: int,
    end: int,
    video_meta: dict[str, str],
    disagreement: dict[str, dict[str, str]],
    overlap_clips: int,
) -> int:
    records = records_for_part_window(rows, start, end, video_meta, disagreement, 0, overlap_clips)
    prompt, _ = prompt_for_records(video_meta, records)
    return len(prompt)


def split_main_ranges(
    rows: list[dict[str, str]],
    video_meta: dict[str, str],
    disagreement: dict[str, dict[str, str]],
    target_chars: int,
    hard_max_chars: int,
    overlap_clips: int,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    ranges: list[tuple[int, int]] = []
    failures: list[dict[str, object]] = []
    start = 0
    while start < len(rows):
        end = start
        last_good = None
        while end < len(rows):
            main_chars = range_prompt_chars(rows, start, end + 1, video_meta, disagreement, 0)
            if main_chars > hard_max_chars and end == start:
                failures.append(
                    {
                        "video_id": video_meta["video_id"],
                        "clip_id": rows[start].get("clip_id", ""),
                        "reason": "failed_context_too_large",
                        "estimated_chars": main_chars,
                        "notes": "single_clip_exceeds_hard_max",
                    }
                )
                start += 1
                break

            desired_chars = range_prompt_chars(rows, start, end + 1, video_meta, disagreement, overlap_clips)
            if desired_chars <= hard_max_chars:
                if desired_chars <= target_chars or last_good is None:
                    last_good = end + 1
                    end += 1
                    continue
                break
            if last_good is None and main_chars <= hard_max_chars:
                last_good = end + 1
            break
        else:
            if last_good is not None:
                ranges.append((start, last_good))
                start = last_good
            continue
        if last_good is not None and start < len(rows):
            ranges.append((start, last_good))
            start = last_good
    return ranges, failures


def records_for_part(
    rows: list[dict[str, str]],
    start: int,
    end: int,
    video_meta: dict[str, str],
    disagreement: dict[str, dict[str, str]],
    part_index: int | None,
    overlap_clips: int,
    hard_max_chars: int | None = None,
) -> list[dict[str, object]]:
    fallback_records: list[dict[str, object]] = []
    for overlap in range(max(0, overlap_clips), -1, -1):
        records = records_for_part_window(rows, start, end, video_meta, disagreement, part_index, overlap)
        fallback_records = records
        if hard_max_chars is None:
            return records
        prompt, _ = prompt_for_records(video_meta, records)
        if len(prompt) <= hard_max_chars:
            return records
    return fallback_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["browser", "api"], default="api" if os.environ.get("OPENAI_API_KEY") else "browser")
    parser.add_argument("--browser-target-chars", type=int, default=160_000)
    parser.add_argument("--browser-hard-max-chars", type=int, default=220_000)
    parser.add_argument("--api-hard-max-chars", type=int, default=int(os.environ.get("API_JOB_HARD_MAX_CHARS", "500000")))
    parser.add_argument("--overlap-clips", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "api":
        target_chars = min(args.api_hard_max_chars, int(os.environ.get("API_JOB_TARGET_CHARS", str(args.api_hard_max_chars))))
        hard_max_chars = args.api_hard_max_chars
    else:
        target_chars = args.browser_target_chars
        hard_max_chars = args.browser_hard_max_chars
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    disagreement = disagreement_by_clip()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible_rows():
        grouped[video_id_for(row)].append(row)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index_rows = []
    queue_rows = []
    previous_queue = {
        row.get("job_id") or job_name(row.get("video_id", "")): row
        for row in read_csv(JOB_QUEUE)
    }
    previous_main_clip_ids = {
        job_id: main_clip_ids_from_input(OUT / f"{job_id}_input.jsonl")
        for job_id in previous_queue
    }
    current_main_clip_ids: dict[str, set[str]] = {}
    split_failures: list[dict[str, object]] = []
    videos_full_jobs = 0
    videos_split_jobs = 0
    clips_covered_by_split = 0
    for video_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=clip_sort_key)
        first = rows[0]
        video_meta = {
            "video_id": video_id,
            "dataset_group": first.get("dataset_group", ""),
            "source_id": first.get("source_id", ""),
            "title": first.get("titulo", ""),
            "channel": first.get("spk", ""),
            "source_url": first.get("source_url", ""),
        }
        full_job = job_name(video_id)
        full_records = [build_record(row, disagreement, full_job, context_only=False) for row in rows]
        full_prompt, full_input_lines = prompt_for_records(video_meta, full_records)
        if len(full_prompt) <= hard_max_chars:
            job_specs = [(full_job, None, 1, "full", full_records, full_prompt, full_input_lines, len(rows), 0)]
            videos_full_jobs += 1
        else:
            ranges, failures = split_main_ranges(
                rows,
                video_meta,
                disagreement,
                target_chars,
                hard_max_chars,
                args.overlap_clips,
            )
            split_failures.extend(failures)
            job_specs = []
            for part_index, (start, end) in enumerate(ranges):
                part_records = records_for_part(
                    rows,
                    start,
                    end,
                    video_meta,
                    disagreement,
                    part_index,
                    args.overlap_clips,
                    hard_max_chars,
                )
                part_prompt, part_lines = prompt_for_records(video_meta, part_records)
                main_count = sum(1 for record in part_records if not record.get("context_only"))
                context_count = len(part_records) - main_count
                job_specs.append(
                    (
                        job_name(video_id, part_index),
                        part_index,
                        len(ranges),
                        "part",
                        part_records,
                        part_prompt,
                        part_lines,
                        main_count,
                        context_count,
                    )
                )
            if job_specs:
                videos_split_jobs += 1
                clips_covered_by_split += sum(spec[7] for spec in job_specs)

        for jid, part_index, part_count, job_kind, records, prompt, input_lines, main_count, context_count in job_specs:
            safe = safe_name(video_id)
            input_path = OUT / f"{jid}_input.jsonl"
            prompt_path = OUT / f"{jid}_prompt.md"
            raw_path = raw_dir / f"{jid}_raw.jsonl"
            current_main_clip_ids[jid] = {
                str(record.get("clip_id", ""))
                for record in records
                if not truthy(record.get("context_only"))
            }
            input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8", newline="\n")
            prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
            estimated_chars = len(prompt)
            notes = ""
            if estimated_chars > hard_max_chars:
                notes = "job_exceeds_hard_max"
            row_common = {
                "job_id": jid,
                "video_id": video_id,
                "safe_video_id": safe,
                "part_index": "" if part_index is None else part_index,
                "part_count": part_count,
                "job_kind": job_kind,
                "dataset_group": first.get("dataset_group", ""),
                "source_id": first.get("source_id", ""),
                "title": first.get("titulo", ""),
                "channel": first.get("spk", ""),
                "source_url": first.get("source_url", ""),
                "clips_eligible": main_count,
                "main_clips": main_count,
                "context_clips": context_count,
                "estimated_chars": estimated_chars,
                "prompt_path": str(prompt_path),
                "input_jsonl_path": str(input_path),
                "raw_output_path": str(raw_path),
                "status": "pending",
                "created_at": now,
                "notes": notes,
            }
            index_rows.append(row_common)
            queue_rows.append(
                {
                    **{field: row_common.get(field, "") for field in QUEUE_FIELDS},
                    "validated_path": "",
                    "applied_at": "",
                    "chat_url": "",
                    "error": "",
                    "updated_at": now,
                }
            )

    write_csv(JOB_INDEX, index_rows, INDEX_FIELDS)
    write_csv(CONTEXT_FAILURES, split_failures, FAILURE_FIELDS)
    merged_queue = []
    for row in queue_rows:
        previous = previous_queue.get(row["job_id"], {})
        same_main_clips = previous_main_clip_ids.get(row["job_id"]) == current_main_clip_ids.get(row["job_id"])
        if same_main_clips and previous.get("status") in {"completed_raw_saved", "validated", "applied"}:
            row.update({k: previous.get(k, row.get(k, "")) for k in QUEUE_FIELDS})
        merged_queue.append(row)
    merged_queue.sort(key=lambda row: int(row.get("estimated_chars") or 0))
    write_csv(JOB_QUEUE, merged_queue, QUEUE_FIELDS)

    counts = Counter(row["dataset_group"] for row in index_rows)
    SPLIT_REPORT.write_text(
        "\n".join(
            [
                "# GPT video job split report",
                "",
                f"mode: {args.mode}",
                f"target_chars: {target_chars}",
                f"hard_max_chars: {hard_max_chars}",
                f"overlap_clips: {args.overlap_clips}",
                f"videos_full_jobs: {videos_full_jobs}",
                f"videos_split_jobs: {videos_split_jobs}",
                f"total_parts: {sum(1 for r in index_rows if r.get('job_kind') == 'part')}",
                f"clips_covered_by_split: {clips_covered_by_split}",
                f"clips_failed_context_too_large: {len(split_failures)}",
                f"total_jobs: {len(index_rows)}",
                f"total_main_clips: {sum(int(r['main_clips']) for r in index_rows)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"video_jobs={len(index_rows)} clips={sum(int(r['main_clips']) for r in index_rows)}")
    print(f"dataset_group_counts={dict(counts)}")
    print(f"videos_full_jobs={videos_full_jobs} videos_split_jobs={videos_split_jobs} total_parts={sum(1 for r in index_rows if r.get('job_kind') == 'part')}")
    print(f"clips_failed_context_too_large={len(split_failures)}")
    print(f"index -> {JOB_INDEX}")
    print(f"queue -> {JOB_QUEUE}")
    print(f"split_report -> {SPLIT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
