"""Genera resume_plan y pack de revision humana para full-clean-release."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_release"
DISCOVERY = ROOT / "data_discovery" / "outputs"
PACK_DIR = OUT_DIR / "human_review_pack"

TARGET_LARGE_TURBO = 12000

FINAL = OUT_DIR / "final_release_manifest.csv"
SOURCE_MAPPING = OUT_DIR / "source_mapping.csv"
ALIGNMENT = OUT_DIR / "alignment_manifest.csv"
FAILURES = OUT_DIR / "reports" / "failures.csv"
SHORTLIST = DISCOVERY / "shortlist_recommended.csv"
INGEST_PLAN = DISCOVERY / "ingest_plan_v1.md"
NEW_INGEST = OUT_DIR / "new_discovery_ingest_manifest.csv"
RESUME_PLAN = OUT_DIR / "resume_plan.md"

SOURCE_FIELDS = [
    "source_id",
    "spk",
    "titulo",
    "clip_count",
    "current_match_confidence",
    "current_candidate_url",
    "current_candidate_title",
    "current_candidate_channel",
    "failure_reason",
    "example_clip_ids",
    "example_texts",
    "youtube_search_query_1",
    "youtube_search_query_2",
    "youtube_search_query_3",
    "google_search_query",
    "manual_url",
    "manual_confidence",
    "manual_notes",
]

NEW_FIELDS = [
    "url",
    "video_id",
    "title",
    "channel",
    "decision",
    "accepted_clips_estimate",
    "usable_minutes_estimate",
    "ingest_status",
    "failure_reason",
    "manual_alternative_url",
    "manual_download_path",
    "manual_notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def clean_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:180]


def video_id_from_url(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", url or "")
    return match.group(1) if match else ""


def source_summary() -> tuple[list[dict[str, object]], list[str]]:
    sources = {row["source_id"]: row for row in read_csv(SOURCE_MAPPING)}
    final = read_csv(FINAL)
    alignment = read_csv(ALIGNMENT)
    failures = read_csv(FAILURES)

    by_source_final: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in final:
        by_source_final[row.get("source_id", "")].append(row)

    by_source_alignment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in alignment:
        by_source_alignment[row.get("source_id", "")].append(row)

    failure_by_source: dict[str, list[str]] = defaultdict(list)
    for row in failures:
        sid = row.get("source_id", "")
        if sid:
            msg = row.get("error_type") or row.get("failure_reason") or row.get("error_message")
            if msg:
                failure_by_source[sid].append(msg)

    rows: list[dict[str, object]] = []
    commands: list[str] = []
    for sid, source in sources.items():
        final_rows = by_source_final.get(sid, [])
        if not final_rows:
            continue
        clean_counts = Counter(row.get("clean_status", "") for row in final_rows)
        align_rows = by_source_alignment.get(sid, [])
        low_score = any(float(row.get("alignment_score") or 0.0) < 70.0 for row in align_rows if row.get("alignment_score"))
        clip_count = int(float(source.get("clip_count") or len(final_rows) or 0))
        blocked_count = clean_counts.get("blocked_source_not_found", 0) + clean_counts.get("blocked_alignment_failed", 0)
        include = (
            source.get("match_confidence") in {"low", "none"}
            or blocked_count > 0
            or low_score
            or (clip_count >= 100 and clean_counts.get("completed_large_turbo_no_gpt", 0) == 0)
        )
        if not include:
            continue

        example_texts = source.get("text_examples", "")
        title = source.get("titulo", "")
        spk = source.get("spk", "")
        failure_bits = []
        if source.get("match_confidence") in {"low", "none"}:
            failure_bits.append(f"match_confidence={source.get('match_confidence')}")
        for status in ("blocked_source_not_found", "blocked_alignment_failed", "baseline_existing_only"):
            if clean_counts.get(status):
                failure_bits.append(f"{status}={clean_counts[status]}")
        if low_score:
            failure_bits.append("low_alignment_score_present")
        failure_bits.extend(sorted(set(failure_by_source.get(sid, [])))[:3])

        rows.append(
            {
                "source_id": sid,
                "spk": spk,
                "titulo": title,
                "clip_count": clip_count,
                "current_match_confidence": source.get("match_confidence", ""),
                "current_candidate_url": source.get("candidate_url", ""),
                "current_candidate_title": source.get("candidate_title", ""),
                "current_candidate_channel": source.get("candidate_channel", ""),
                "failure_reason": "|".join(failure_bits),
                "example_clip_ids": source.get("example_clips", ""),
                "example_texts": example_texts,
                "youtube_search_query_1": clean_query(f"{title}"),
                "youtube_search_query_2": clean_query(f"{spk} {title}"),
                "youtube_search_query_3": clean_query((example_texts.split(" || ")[0] if example_texts else title)),
                "google_search_query": clean_query(f"site:youtube.com {title}"),
                "manual_url": "",
                "manual_confidence": "",
                "manual_notes": "",
            }
        )
        commands.append(f"python data_release/scripts/reconstruct_existing_clips.py --source {sid} --resume --upload --checkpoint-every 25")

    rows.sort(key=lambda row: int(row["clip_count"]), reverse=True)
    return rows, commands


def new_discovery_rows() -> list[dict[str, object]]:
    ingest = {row.get("url", ""): row for row in read_csv(NEW_INGEST)}
    out: list[dict[str, object]] = []
    for row in read_csv(SHORTLIST):
        if row.get("decision") not in {"strong_accept", "accept"}:
            continue
        current = ingest.get(row.get("url", ""), {})
        video_id = row.get("video_id") or video_id_from_url(row.get("url", ""))
        out.append(
            {
                "url": row.get("url", ""),
                "video_id": video_id,
                "title": row.get("title", ""),
                "channel": row.get("channel", ""),
                "decision": row.get("decision", ""),
                "accepted_clips_estimate": row.get("accepted_clips_estimate", ""),
                "usable_minutes_estimate": row.get("usable_minutes_estimate", ""),
                "ingest_status": current.get("ingest_status", "blocked_download_failed"),
                "failure_reason": current.get("failure_reason", "youtube_requires_login_or_cookies_from_vm"),
                "manual_alternative_url": "",
                "manual_download_path": "",
                "manual_notes": "",
            }
        )
    out.sort(key=lambda row: float(row.get("accepted_clips_estimate") or 0.0), reverse=True)
    return out


def write_readme(source_rows: list[dict[str, object]], new_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Human review pack",
        "",
        "Este pack permite destrabar fuentes sin tocar columnas automaticas.",
        "",
        "## Archivos",
        "",
        "- `human_source_mapping_needed.csv`: completar `manual_url`, `manual_confidence` y `manual_notes` para fuentes existing.",
        "- `human_new_discovery_download_needed.csv`: completar `manual_alternative_url` o `manual_download_path` para accepted new_discovery que no descargan automaticamente.",
        "",
        "## Reglas",
        "",
        "- No editar columnas automaticas.",
        "- `manual_confidence` debe ser `high`, `medium` o `low`.",
        "- `manual_download_path` debe apuntar a un archivo local descargado fuera del repo o en un directorio ignorado.",
        "- No guardar cookies, tokens ni credenciales en este directorio.",
        "",
        "## Como reanudar despues de completar el CSV",
        "",
        "1. Descargar fuentes localmente con:",
        "",
        "```powershell",
        "python data_release/scripts/download_sources_local.py --new-discovery --limit 5",
        "```",
        "",
        "2. Para existing con `manual_url`, reintentar reconstruccion desde una fuente ya descargada/subida a GCS.",
        "",
        "```powershell",
        "python data_release/scripts/download_sources_local.py --existing --limit 5",
        "```",
        "",
        "3. Crear VM GPU solo cuando haya fuentes en GCS listas para segmentar/ASR.",
        "",
        "## Top existing a revisar",
        "",
    ]
    for row in source_rows[:20]:
        lines.append(f"- {row['source_id']} clips={row['clip_count']} reason={row['failure_reason']}")
    lines.extend(["", "## Top new_discovery a descargar", ""])
    for row in new_rows[:20]:
        lines.append(f"- {row['video_id']} clips={row['accepted_clips_estimate']} {row['title']}")
    lines.append("")
    (PACK_DIR / "README_HUMAN_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def write_resume_plan(source_rows: list[dict[str, object]], new_rows: list[dict[str, object]], commands: list[str]) -> None:
    final = read_csv(FINAL)
    clean_counts = Counter(row.get("clean_status", "") for row in final)
    asr_counts = Counter(row.get("asr_status", "") for row in final)
    completed_large_turbo = clean_counts.get("completed_large_turbo_no_gpt", 0) + clean_counts.get("completed_clean_gpt", 0)
    usable = sum(1 for row in final if row.get("usable_for_training") == "true")
    estimated_new = sum(float(row.get("accepted_clips_estimate") or 0.0) for row in new_rows)
    lines = [
        "# Resume plan",
        "",
        f"target_large_turbo_clips: {TARGET_LARGE_TURBO}",
        f"completed_large_turbo_or_clean_gpt: {completed_large_turbo}",
        f"remaining_to_12k_large_turbo: {max(0, TARGET_LARGE_TURBO - completed_large_turbo)}",
        f"usable_for_training_total_in_manifest: {usable}",
        f"remaining_to_12k_usable: {max(0, TARGET_LARGE_TURBO - usable)}",
        f"baseline_existing_only: {clean_counts.get('baseline_existing_only', 0)}",
        f"completed_clean_gpt: {clean_counts.get('completed_clean_gpt', 0)}",
        f"completed_large_turbo_no_gpt: {clean_counts.get('completed_large_turbo_no_gpt', 0)}",
        f"asr_status_counts: {dict(sorted(asr_counts.items()))}",
        "",
        "## New discovery priority",
        "",
        f"accepted_or_strong_accept_videos: {len(new_rows)}",
        f"estimated_new_accepted_clips: {round(estimated_new, 1)}",
        "blocker: youtube_requires_login_or_cookies_from_vm; use local download flow, never upload cookies.",
        "",
        "## Next new_discovery videos",
        "",
    ]
    for row in new_rows[:20]:
        lines.append(f"- {row['video_id']} clips={row['accepted_clips_estimate']} score_source={row['decision']} url={row['url']}")
    lines.extend(["", "## Existing manual review priority", ""])
    for row in source_rows[:20]:
        lines.append(f"- {row['source_id']} clips={row['clip_count']} confidence={row['current_match_confidence']} reason={row['failure_reason']}")
    lines.extend(["", "## Exact commands", ""])
    lines.extend(
        [
            "```powershell",
            "python data_release/scripts/build_resume_review_pack.py",
            "python data_release/scripts/download_sources_local.py --new-discovery --limit 5",
            "python data_release/scripts/download_sources_local.py --existing --limit 5",
            "python data_release/scripts/build_full_clean_release_outputs.py",
            "python data_release/scripts/validate_clean_bucket.py",
            "```",
            "",
            "## Existing VM commands after sources are in GCS",
            "",
            "```bash",
        ]
    )
    lines.extend(commands[:20])
    lines.extend(["```", ""])
    if INGEST_PLAN.exists():
        lines.extend(["## Ingest plan", "", f"See `{INGEST_PLAN.as_posix()}`.", ""])
    RESUME_PLAN.write_text("\n".join(lines), encoding="utf-8")


def write_summary(source_rows: list[dict[str, object]], new_rows: list[dict[str, object]]) -> None:
    report = OUT_DIR / "reports" / "human_review_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Human review summary",
        "",
        f"existing_sources_to_review: {len(source_rows)}",
        f"new_discovery_downloads_to_review: {len(new_rows)}",
        "",
        "## Top 20 existing",
        "",
    ]
    for row in source_rows[:20]:
        lines.append(f"- {row['source_id']} clips={row['clip_count']} reason={row['failure_reason']}")
    lines.extend(["", "## Top 20 new_discovery", ""])
    for row in new_rows[:20]:
        lines.append(f"- {row['video_id']} clips={row['accepted_clips_estimate']} {row['title']}")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    source_rows, commands = source_summary()
    new_rows = new_discovery_rows()
    write_csv(PACK_DIR / "human_source_mapping_needed.csv", source_rows, SOURCE_FIELDS)
    write_csv(PACK_DIR / "human_new_discovery_download_needed.csv", new_rows, NEW_FIELDS)
    write_readme(source_rows, new_rows)
    write_resume_plan(source_rows, new_rows, commands)
    write_summary(source_rows, new_rows)
    print(f"source_review_rows={len(source_rows)} -> {PACK_DIR / 'human_source_mapping_needed.csv'}")
    print(f"new_download_rows={len(new_rows)} -> {PACK_DIR / 'human_new_discovery_download_needed.csv'}")
    print(f"resume_plan -> {RESUME_PLAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
