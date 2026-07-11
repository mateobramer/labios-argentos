"""Construye manifests livianos para el bucket limpio v1.

No descarga videos, audios ni ROIs. Solo lee listados de GCS y CSV/textos chicos
necesarios para armar trazabilidad.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SOURCE_BUCKET = "gs://labios-argentos-vsr-dataset"
LIP_ROIS_PREFIX = f"{SOURCE_BUCKET}/lip_rois/"
CURRICULUM_PREFIX = f"{SOURCE_BUCKET}/curriculum_visper/"
SPLITS_CSV = f"{SOURCE_BUCKET}/splits/splits.csv"
CANDIDATOS_CSV = f"{SOURCE_BUCKET}/config/candidatos_v2_FINAL.csv"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_pipeline/release"
CACHE_DIR = OUT_DIR / "cache"
REPORTS_DIR = OUT_DIR / "reports"

ARG_EXISTING = OUT_DIR / "argentina_existing_manifest.csv"
SPANISH_GENERAL = OUT_DIR / "spanish_general_manifest.csv"
ARG_NEW = OUT_DIR / "argentina_new_manifest.csv"
MANIFEST_REPORT = REPORTS_DIR / "manifest_build_report.md"
FAILURES = REPORTS_DIR / "failures.csv"
BUCKET_BUILD_REPORT = REPORTS_DIR / "bucket_build_report.md"
COST_RUNTIME_REPORT = REPORTS_DIR / "cost_runtime_report.md"

SHORTLIST = ROOT / "data_pipeline/discovery" / "outputs" / "shortlist_recommended.csv"
CANDIDATE_SCORES = ROOT / "data_pipeline/discovery" / "outputs" / "candidate_scores.csv"
TARGET_PROGRESS = ROOT / "data_pipeline/discovery" / "outputs" / "target_progress.md"
BUCKET_INVENTORY = ROOT / "data_pipeline/inventory" / "bucket_inventory.csv"


ARG_EXISTING_FIELDS = [
    "dataset_group",
    "source_bucket",
    "source_type",
    "clip_id",
    "split",
    "spk",
    "titulo",
    "clip",
    "n_frames",
    "text_large_existing",
    "mp4_gcs_path",
    "npz_gcs_path",
    "txt_gcs_path",
    "has_mp4",
    "has_npz",
    "has_txt",
    "url",
    "url_confidence",
    "source_context_status",
    "notes",
]

SPANISH_FIELDS = [
    "dataset_group",
    "source_bucket",
    "clip_id",
    "mp4_gcs_path",
    "npz_gcs_path",
    "txt_gcs_path",
    "has_mp4",
    "has_npz",
    "has_txt",
    "text_large_existing",
    "text_turbo_path",
    "license_status",
    "provenance_status",
    "usable_status",
    "notes",
]

ARG_NEW_FIELDS = [
    "url",
    "video_id",
    "title",
    "channel",
    "decision",
    "total_score",
    "visual_quality_score",
    "audio_quality_score",
    "usable_minutes_estimate",
    "accepted_clips_estimate",
    "recommended_use",
    "source_type",
    "expected_accent",
    "ingest_status",
    "notes",
]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not found:
        raise RuntimeError(f"No se encontro {name} en PATH")
    return found


GSUTIL = tool("gsutil")


def run_bytes(args: list[str], timeout: int = 900) -> bytes:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"Fallo comando: {' '.join(args)}\n{stderr}")
    return result.stdout


def cached_bytes(cache_name: str, args: list[str], refresh: bool = False, timeout: int = 900) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    data = run_bytes(args, timeout=timeout)
    cache_path.write_bytes(data)
    return data


def gsutil_cat(path: str, cache_name: str, refresh: bool = False) -> str:
    data = cached_bytes(cache_name, [GSUTIL, "cat", path], refresh=refresh, timeout=180)
    return data.decode("utf-8", "replace")


def gsutil_list(prefix: str, cache_name: str, refresh: bool = False) -> list[str]:
    data = cached_bytes(cache_name, [GSUTIL, "ls", "-r", prefix + "**"], refresh=refresh, timeout=1200)
    text = data.decode("utf-8", "replace")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("gs://") and not line.strip().endswith(":")
    ]


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def read_local_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def split_spk_to_n(spk: str) -> str:
    match = re.fullmatch(r"f0*(\d+)", (spk or "").strip().casefold())
    return str(int(match.group(1))) if match else ""


def make_url_mapper(candidatos: list[dict[str, str]]):
    by_n: dict[str, dict[str, str]] = {}
    by_hablante: list[tuple[str, dict[str, str]]] = []
    for row in candidatos:
        n = str(row.get("n", "")).strip()
        if n and n not in by_n:
            by_n[n] = row
        hablante_norm = norm(row.get("hablante", ""))
        if hablante_norm:
            by_hablante.append((hablante_norm, row))

    def mapper(spk: str, titulo: str) -> tuple[str, str, str]:
        n = split_spk_to_n(spk)
        if n in by_n and by_n[n].get("url"):
            return by_n[n]["url"], "spk_n_match", "url_mapped_from_config_candidates"
        titulo_norm = norm(titulo)
        for hablante_norm, row in by_hablante:
            if hablante_norm and hablante_norm in titulo_norm and row.get("url"):
                return row["url"], "title_hablante_match", "url_mapped_from_config_candidates"
        return "", "", "missing_url"

    return mapper


def remap_npz(npz_path: str) -> str:
    prefix = "data/processed/lip_rois/"
    if npz_path.startswith(prefix):
        return LIP_ROIS_PREFIX + npz_path[len(prefix) :]
    return LIP_ROIS_PREFIX + npz_path.replace("\\", "/").split("lip_rois/", 1)[-1]


def build_argentina_existing(refresh: bool = False) -> tuple[list[dict[str, object]], dict[str, object]]:
    splits = read_csv_text(gsutil_cat(SPLITS_CSV, "splits.csv", refresh=refresh))
    candidatos = read_csv_text(gsutil_cat(CANDIDATOS_CSV, "candidatos_v2_FINAL.csv", refresh=refresh))
    files = set(gsutil_list(LIP_ROIS_PREFIX, "lip_rois_listing.txt", refresh=refresh))
    mapper = make_url_mapper(candidatos)

    rows: list[dict[str, object]] = []
    expected_files: set[str] = set()
    split_counts: Counter[str] = Counter()
    url_counts: Counter[str] = Counter()

    for row in splits:
        split_counts[row.get("split", "")] += 1
        npz = remap_npz(row.get("npz", ""))
        stem = npz.rsplit(".", 1)[0]
        mp4 = stem + ".mp4"
        txt = stem + ".txt"
        expected_files.update([mp4, npz, txt])
        url, confidence, context_status = mapper(row.get("spk", ""), row.get("titulo", ""))
        url_counts["mapped" if url else "missing"] += 1
        has_mp4 = mp4 in files
        has_npz = npz in files
        has_txt = txt in files
        notes = []
        if not (has_mp4 and has_npz and has_txt):
            notes.append("missing_triplet_member")
        rows.append(
            {
                "dataset_group": "argentina/existing",
                "source_bucket": SOURCE_BUCKET,
                "source_type": "existing_lip_rois_splits",
                "clip_id": f"{row.get('spk','')}::{row.get('titulo','')}::{row.get('clip','')}",
                "split": row.get("split", ""),
                "spk": row.get("spk", ""),
                "titulo": row.get("titulo", ""),
                "clip": row.get("clip", ""),
                "n_frames": row.get("n_frames", ""),
                "text_large_existing": row.get("texto", ""),
                "mp4_gcs_path": mp4,
                "npz_gcs_path": npz,
                "txt_gcs_path": txt,
                "has_mp4": str(has_mp4).lower(),
                "has_npz": str(has_npz).lower(),
                "has_txt": str(has_txt).lower(),
                "url": url,
                "url_confidence": confidence,
                "source_context_status": context_status,
                "notes": "|".join(notes),
            }
        )

    unused_files = files - expected_files
    stats = {
        "rows": len(rows),
        "triplets_found": sum(
            1 for r in rows if r["has_mp4"] == "true" and r["has_npz"] == "true" and r["has_txt"] == "true"
        ),
        "rows_missing_any_file": sum(
            1 for r in rows if not (r["has_mp4"] == "true" and r["has_npz"] == "true" and r["has_txt"] == "true")
        ),
        "bucket_files": len(files),
        "bucket_files_unused_by_splits": len(unused_files),
        "split_counts": dict(split_counts),
        "url_counts": dict(url_counts),
        "unused_examples": sorted(unused_files)[:10],
    }
    return rows, stats


def build_spanish_general(refresh: bool = False) -> tuple[list[dict[str, object]], dict[str, object]]:
    files = gsutil_list(CURRICULUM_PREFIX, "curriculum_visper_listing.txt", refresh=refresh)
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    ext_counts: Counter[str] = Counter()
    for path in files:
        if "." not in path.rsplit("/", 1)[-1]:
            continue
        stem, ext = path.rsplit(".", 1)
        ext = "." + ext.lower()
        ext_counts[ext] += 1
        if ext in {".mp4", ".npz", ".txt"}:
            grouped[stem][ext] = path

    rows: list[dict[str, object]] = []
    for stem, items in sorted(grouped.items()):
        txt = items.get(".txt", "")
        rel = stem.replace(CURRICULUM_PREFIX, "", 1)
        has_mp4 = ".mp4" in items
        has_npz = ".npz" in items
        has_txt = ".txt" in items
        notes = []
        if not has_npz:
            notes.append("missing_npz")
        if not has_mp4:
            notes.append("missing_mp4")
        if not has_txt:
            notes.append("missing_txt")
        if txt:
            notes.append("large_existing_available_in_txt_gcs_path")
        rows.append(
            {
                "dataset_group": "spanish_general/existing",
                "source_bucket": SOURCE_BUCKET,
                "clip_id": rel,
                "mp4_gcs_path": items.get(".mp4", ""),
                "npz_gcs_path": items.get(".npz", ""),
                "txt_gcs_path": txt,
                "has_mp4": str(has_mp4).lower(),
                "has_npz": str(has_npz).lower(),
                "has_txt": str(has_txt).lower(),
                "text_large_existing": txt,
                "text_turbo_path": "",
                "license_status": "unknown_or_sensitive",
                "provenance_status": "not_documented_in_bucket",
                "usable_status": "prepared_separate_not_argentina",
                "notes": "|".join(notes),
            }
        )

    stats = {
        "rows": len(rows),
        "files": len(files),
        "extensions": dict(ext_counts),
        "complete_triplets": sum(
            1 for r in rows if r["has_mp4"] == "true" and r["has_npz"] == "true" and r["has_txt"] == "true"
        ),
    }
    return rows, stats


def build_argentina_new() -> tuple[list[dict[str, object]], dict[str, object]]:
    shortlist = read_local_csv(SHORTLIST)
    scores = read_local_csv(CANDIDATE_SCORES)
    by_url = {row.get("url", ""): row for row in scores if row.get("url")}
    rows: list[dict[str, object]] = []
    for row in shortlist:
        if row.get("decision") not in {"strong_accept", "accept"}:
            continue
        score = by_url.get(row.get("url", ""), {})
        rows.append(
            {
                "url": row.get("url", ""),
                "video_id": score.get("video_id", ""),
                "title": row.get("title", ""),
                "channel": row.get("channel", ""),
                "decision": row.get("decision", ""),
                "total_score": row.get("total_score", ""),
                "visual_quality_score": row.get("visual_quality_score", ""),
                "audio_quality_score": row.get("audio_quality_score", ""),
                "usable_minutes_estimate": row.get("usable_minutes_estimate", ""),
                "accepted_clips_estimate": row.get("accepted_clips_estimate", ""),
                "recommended_use": row.get("recommended_use", ""),
                "source_type": score.get("source_type", ""),
                "expected_accent": score.get("expected_accent", ""),
                "ingest_status": "pending_vm_processing",
                "notes": "accepted_only_from_data_pipeline/discovery",
            }
        )
    return rows, {"rows": len(rows), "decisions": dict(Counter(r["decision"] for r in rows))}


def write_initial_reports(stats: dict[str, dict[str, object]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_progress = TARGET_PROGRESS.read_text(encoding="utf-8") if TARGET_PROGRESS.exists() else ""
    inventory_note = "Disponible" if BUCKET_INVENTORY.exists() else "No encontrado"
    lines = [
        "# Manifest build report",
        "",
        f"source_bucket: `{SOURCE_BUCKET}`",
        f"bucket_inventory_csv: {inventory_note}",
        "",
        "## Argentina existing",
        f"- filas en splits: {stats['argentina_existing']['rows']}",
        f"- tripletas encontradas: {stats['argentina_existing']['triplets_found']}",
        f"- filas con algun archivo faltante: {stats['argentina_existing']['rows_missing_any_file']}",
        f"- archivos en bucket no usados por splits: {stats['argentina_existing']['bucket_files_unused_by_splits']}",
        f"- distribucion split: {stats['argentina_existing']['split_counts']}",
        f"- URL mapeada/sin URL: {stats['argentina_existing']['url_counts']}",
        "",
        "## Spanish general",
        f"- filas manifest: {stats['spanish_general']['rows']}",
        f"- archivos listados: {stats['spanish_general']['files']}",
        f"- extensiones: {stats['spanish_general']['extensions']}",
        f"- tripletas completas: {stats['spanish_general']['complete_triplets']}",
        "- licencia/procedencia: unknown_or_sensitive / not_documented_in_bucket",
        "",
        "## Argentina new discovery",
        f"- videos accepted: {stats['argentina_new']['rows']}",
        f"- decisiones: {stats['argentina_new']['decisions']}",
        "",
        "## Target progress snapshot",
        "```",
        target_progress.strip(),
        "```",
        "",
    ]
    MANIFEST_REPORT.write_text("\n".join(lines), encoding="utf-8")

    if not FAILURES.exists():
        write_csv(
            FAILURES,
            [],
            ["stage", "dataset_group", "source_id", "clip_id", "path", "error_type", "error_message", "notes"],
        )

    BUCKET_BUILD_REPORT.write_text(
        "\n".join(
            [
                "# Bucket build report",
                "",
                "bucket_destino: gs://labios-argentos-vsr-clean-v1/",
                "estado: initialized_local_manifests",
                "source_bucket: gs://labios-argentos-vsr-dataset/",
                "no se modifico ni borro el bucket fuente.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    COST_RUNTIME_REPORT.write_text(
        "\n".join(
            [
                "# Cost/runtime report",
                "",
                "project: labios-argentos-499900",
                "bucket_destino: gs://labios-argentos-vsr-clean-v1/",
                "vm_status: not_created_yet",
                "gpu_plan: prefer nvidia-l4, fallback nvidia-tesla-t4",
                "cleanup_plan: delete VM, auto-delete boot disk, verify no static IPs/disks.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    refresh = "--refresh" in argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_rows, existing_stats = build_argentina_existing(refresh=refresh)
    write_csv(ARG_EXISTING, existing_rows, ARG_EXISTING_FIELDS)

    spanish_rows, spanish_stats = build_spanish_general(refresh=refresh)
    write_csv(SPANISH_GENERAL, spanish_rows, SPANISH_FIELDS)

    new_rows, new_stats = build_argentina_new()
    write_csv(ARG_NEW, new_rows, ARG_NEW_FIELDS)

    write_initial_reports(
        {
            "argentina_existing": existing_stats,
            "spanish_general": spanish_stats,
            "argentina_new": new_stats,
        }
    )
    print(f"argentina_existing_rows={existing_stats['rows']} -> {ARG_EXISTING}")
    print(f"spanish_general_rows={spanish_stats['rows']} -> {SPANISH_GENERAL}")
    print(f"argentina_new_rows={new_stats['rows']} -> {ARG_NEW}")
    print(f"report -> {MANIFEST_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
