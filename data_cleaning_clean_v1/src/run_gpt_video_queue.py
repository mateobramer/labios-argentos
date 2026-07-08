"""Corre cola concurrente de GPT cleaning por video_id completo."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data_cleaning_clean_v1"
QUEUE = ROOT / "video_jobs" / "video_job_queue.csv"
MANUAL = ROOT / "video_jobs" / "manual_browser_queue.md"

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

TERMINAL = {
    "completed_raw_saved",
    "validated",
    "applied",
    "failed_browser",
    "failed_jsonl",
    "failed_rate_limit",
    "failed_context_too_large",
    "skipped_no_eligible_clips",
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def row_key(row: dict[str, str]) -> str:
    return row.get("job_id") or f"video_{row.get('safe_video_id') or row.get('video_id', '')}"


def update_row(rows: list[dict[str, str]], job_id: str, **updates: object) -> None:
    for row in rows:
        if row_key(row) == job_id:
            for key, value in updates.items():
                row[key] = str(value)
            row["updated_at"] = now()
            return


def extract_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") in {"output_text", "text"} and isinstance(value.get("text"), str):
                parts.append(value["text"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(response.get("output", []))
    return "\n".join(part for part in parts if part).strip()


def call_openai(prompt: str, model: str, timeout: int) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    payload: dict[str, object] = {
        "model": model,
        "input": prompt,
    }
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "high").strip()
    if effort and effort.lower() != "none":
        payload["reasoning"] = {"effort": effort}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = extract_text(data)
    if not text:
        raise RuntimeError("empty_model_output")
    return text


def run_one(row: dict[str, str], model: str, max_prompt_chars: int, timeout: int) -> dict[str, str]:
    job_id = row_key(row)
    prompt_path = Path(row["prompt_path"])
    raw_path = Path(row["raw_output_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    if len(prompt) > max_prompt_chars:
        return {"job_id": job_id, "status": "failed_context_too_large", "error": f"chars={len(prompt)}"}
    try:
        text = call_openai(prompt, model, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-1000:]
        if exc.code == 429:
            return {"job_id": job_id, "status": "failed_rate_limit", "error": body}
        if exc.code == 400 and ("context" in body.lower() or "token" in body.lower()):
            return {"job_id": job_id, "status": "failed_context_too_large", "error": body}
        return {"job_id": job_id, "status": "failed_jsonl", "error": body}
    except Exception as exc:  # noqa: BLE001
        return {"job_id": job_id, "status": "failed_jsonl", "error": str(exc)[-1000:]}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")
    return {"job_id": job_id, "status": "completed_raw_saved", "error": "", "raw_output_path": str(raw_path)}


def validate(job_id: str, video_id: str, raw_path: str) -> bool:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "validate_gpt_video_jsonl.py"),
        "--job-id",
        job_id,
        "--video-id",
        video_id,
        "--raw-path",
        raw_path,
    ]
    return subprocess.run(cmd, cwd=REPO).returncode == 0


def apply(job_id: str) -> bool:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "apply_gpt_video_patches.py"),
        "--job-id",
        job_id,
    ]
    return subprocess.run(cmd, cwd=REPO).returncode == 0


def write_manual_queue(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Manual browser queue",
        "",
        "No hay `OPENAI_API_KEY` en el entorno. Para usar ChatGPT navegador:",
        "",
        "1. Abrir una conversacion por cada video_id.",
        "2. Pegar el contenido del `prompt_path` correspondiente.",
        "3. Guardar la respuesta JSONL exacta en `raw_output_path`.",
        "4. Correr `validate_gpt_video_jsonl.py` y luego `apply_gpt_video_patches.py`.",
        "",
        "| job_id | clips | prompt_path | raw_output_path |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
        f"| {row_key(row)} | {row.get('clips_eligible', '')} | {row.get('prompt_path', '')} | {row.get('raw_output_path', '')} |"
        )
    MANUAL.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-in-flight", type=int, default=int(os.environ.get("MAX_IN_FLIGHT", "4")))
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--max-prompt-chars", type=int, default=int(os.environ.get("MAX_PROMPT_CHARS", "500000")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "900")))
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(QUEUE)
    if not rows:
        print("No existe cola. Correr export_gpt_video_jobs.py primero.")
        return 1
    pending = [row for row in rows if row.get("status") not in TERMINAL]
    pending.sort(key=lambda row: int(row.get("estimated_chars") or 0))
    if args.limit_videos is not None:
        pending = pending[: args.limit_videos]
    if not pending:
        print("No hay jobs pendientes.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        write_manual_queue(pending)
        print(f"OPENAI_API_KEY ausente; cola manual -> {MANUAL}")
        return 2

    max_workers = max(1, min(args.max_in_flight, 6))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for row in pending:
            update_row(rows, row_key(row), status="running", error="")
            futures[pool.submit(run_one, row, args.model, args.max_prompt_chars, args.timeout)] = row
        write_csv(QUEUE, rows, QUEUE_FIELDS)

        for future in as_completed(futures):
            result = future.result()
            source_row = futures[future]
            job_id = result["job_id"]
            video_id = source_row.get("video_id", "")
            status = result["status"]
            raw_path = result.get("raw_output_path", "")
            update_row(rows, job_id, status=status, raw_output_path=raw_path, error=result.get("error", ""))
            write_csv(QUEUE, rows, QUEUE_FIELDS)
            if status != "completed_raw_saved":
                continue
            if not args.no_validate and validate(job_id, video_id, raw_path):
                validated_path = str(ROOT / "validated" / f"{job_id}_validated.jsonl")
                update_row(rows, job_id, status="validated", validated_path=validated_path)
                write_csv(QUEUE, rows, QUEUE_FIELDS)
                if not args.no_apply and apply(job_id):
                    update_row(rows, job_id, status="applied", applied_at=now())
                    write_csv(QUEUE, rows, QUEUE_FIELDS)
            elif not args.no_validate:
                update_row(rows, job_id, status="failed_jsonl", error="validation_failed")
                write_csv(QUEUE, rows, QUEUE_FIELDS)
            time.sleep(1)

    final_counts: dict[str, int] = {}
    for row in rows:
        final_counts[row.get("status", "")] = final_counts.get(row.get("status", ""), 0) + 1
    print(f"queue_counts={final_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
