"""Construye manifests y reportes finales de full clean release.

La limpieza GPT queda conservadora: solo se marca `completed_clean_gpt` si existe
texto validado. En esta corrida no se inventan patches; los clips con ASR real
quedan como `completed_large_turbo_no_gpt`.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_release"
REPORTS = OUT_DIR / "reports"

ARG_EXISTING = OUT_DIR / "argentina_existing_manifest.csv"
ARG_NEW = OUT_DIR / "argentina_new_manifest.csv"
SPANISH = OUT_DIR / "spanish_general_manifest.csv"
SOURCE_MAPPING = OUT_DIR / "source_mapping.csv"
ALIGNMENT = OUT_DIR / "alignment_manifest.csv"
RECON = OUT_DIR / "existing_reconstruction_manifest.csv"
ASR = OUT_DIR / "asr_large_turbo_manifest.csv"
DISAGREEMENT = OUT_DIR / "asr_disagreement_v2.csv"
LOCAL_DOWNLOADS = OUT_DIR / "local_source_download_manifest.csv"

FINAL_RELEASE = OUT_DIR / "final_release_manifest.csv"
FINAL_TRAIN = OUT_DIR / "final_train_manifest_clean_gpt_v1.csv"
FINAL_EVAL = OUT_DIR / "final_eval_manifest_clean_gpt_v1.csv"
CLEAN_GPT = OUT_DIR / "clean_gpt_manifest.csv"
NEW_DISCOVERY = OUT_DIR / "new_discovery_ingest_manifest.csv"
FAILURES = REPORTS / "failures.csv"
FULL_REPORT = REPORTS / "full_clean_release_report.md"
GPT_REPORT = REPORTS / "gpt_cleaning_report.md"
COST_REPORT = REPORTS / "cost_runtime_report.md"
SPANISH_REPORT = OUT_DIR / "spanish_general_asr_manifest.csv"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"

FINAL_FIELDS = [
    "dataset_group",
    "source_id",
    "clip_id",
    "split",
    "spk",
    "titulo",
    "source_url",
    "source_video_id",
    "start_time",
    "end_time",
    "mp4_visual_roi_path",
    "npz_path",
    "clip_with_audio_path",
    "existing_text",
    "large_text",
    "turbo_text",
    "clean_gpt_text",
    "selected_training_text",
    "text_source",
    "clean_status",
    "clean_confidence",
    "patch_count",
    "alignment_confidence",
    "asr_status",
    "gpt_status",
    "usable_for_training",
    "usable_for_eval",
    "needs_review",
    "failure_reason",
    "notes",
]

CLEAN_FIELDS = [
    "dataset_group",
    "source_id",
    "clip_id",
    "existing_text",
    "large_text",
    "turbo_text",
    "clean_text",
    "status",
    "confidence",
    "patch_count",
    "gpt_status",
    "notes",
]

NEW_FIELDS = [
    "url",
    "video_id",
    "title",
    "channel",
    "decision",
    "total_score",
    "usable_minutes_estimate",
    "accepted_clips_estimate",
    "recommended_use",
    "ingest_status",
    "failure_reason",
    "source_video_gcs_path",
    "source_audio_gcs_path",
    "metadata_gcs_path",
    "notes",
]

SPANISH_FIELDS = [
    "dataset_group",
    "clip_id",
    "mp4_gcs_path",
    "npz_gcs_path",
    "txt_gcs_path",
    "turbo_text_path",
    "asr_status",
    "provenance_status",
    "license_status",
    "notes",
]

FAILURE_FIELDS = ["stage", "dataset_group", "source_id", "clip_id", "path", "error_type", "error_message", "notes"]


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


def source_id_for(row: dict[str, str]) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row.get('spk')}__{row.get('titulo')}").strip("_")[:120]


def load_asr() -> dict[tuple[str, str, str], dict[str, str]]:
    out = {}
    for row in read_csv(ASR):
        if row.get("status") == "completed_asr" and row.get("model_role") in {"large", "turbo"}:
            out[(row.get("source_id", ""), row.get("clip_id", ""), row.get("model_role", ""))] = row
    return out


def build_existing_final() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mapping = {row["source_id"]: row for row in read_csv(SOURCE_MAPPING)}
    alignment = {row["clip_id"]: row for row in read_csv(ALIGNMENT)}
    recon = {row["clip_id"]: row for row in read_csv(RECON) if row.get("status") == "completed_reconstructed_audio"}
    asr = load_asr()
    final_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []

    for row in read_csv(ARG_EXISTING):
        sid = source_id_for(row)
        clip_id = row.get("clip_id", "")
        align = alignment.get(clip_id, {})
        source = mapping.get(sid, {})
        rec = recon.get(clip_id, {})
        large = asr.get((sid, clip_id, "large"), {})
        turbo = asr.get((sid, clip_id, "turbo"), {})
        existing_text = row.get("text_large_existing", "")
        large_text = large.get("text", "")
        turbo_text = turbo.get("text", "")
        if large_text and turbo_text:
            clean_status = "completed_large_turbo_no_gpt"
            asr_status = "completed_large_turbo"
            gpt_status = "not_attempted"
            selected = large_text
            text_source = "large"
            needs_review = "false"
            failure_reason = ""
        elif align.get("status") == "blocked_alignment_failed":
            clean_status = "blocked_alignment_failed"
            asr_status = "blocked_alignment_failed"
            gpt_status = "blocked_no_asr_context"
            selected = existing_text
            text_source = "existing_text"
            needs_review = "true"
            failure_reason = "blocked_alignment_failed"
        elif align.get("status") == "blocked_source_not_found" or source.get("match_confidence") in {"none", "low"}:
            clean_status = "blocked_source_not_found"
            asr_status = "blocked_source_not_found"
            gpt_status = "blocked_no_asr_context"
            selected = existing_text
            text_source = "existing_text"
            needs_review = "true"
            failure_reason = "blocked_source_not_found_or_low_confidence"
        else:
            clean_status = "baseline_existing_only"
            asr_status = "pending_reconstruction_or_asr"
            gpt_status = "blocked_no_asr_context"
            selected = existing_text
            text_source = "existing_text"
            needs_review = "true"
            failure_reason = "not_processed_before_spot_preemption"

        final = {
            "dataset_group": "argentina/existing",
            "source_id": sid,
            "clip_id": clip_id,
            "split": row.get("split", ""),
            "spk": row.get("spk", ""),
            "titulo": row.get("titulo", ""),
            "source_url": source.get("candidate_url", align.get("source_url", "")),
            "source_video_id": source.get("candidate_video_id", align.get("source_video_id", "")),
            "start_time": align.get("start_time", ""),
            "end_time": align.get("end_time", ""),
            "mp4_visual_roi_path": row.get("mp4_gcs_path", ""),
            "npz_path": row.get("npz_gcs_path", ""),
            "clip_with_audio_path": rec.get("clip_video_gcs_path", ""),
            "existing_text": existing_text,
            "large_text": large_text,
            "turbo_text": turbo_text,
            "clean_gpt_text": "",
            "selected_training_text": selected,
            "text_source": text_source,
            "clean_status": clean_status,
            "clean_confidence": "medium" if clean_status == "completed_large_turbo_no_gpt" else "low",
            "patch_count": 0,
            "alignment_confidence": align.get("alignment_confidence", ""),
            "asr_status": asr_status,
            "gpt_status": gpt_status,
            "usable_for_training": str(bool(selected)).lower(),
            "usable_for_eval": str(bool(selected) and row.get("split", "") in {"val", "test"}).lower(),
            "needs_review": needs_review,
            "failure_reason": failure_reason,
            "notes": "gpt_cleaning_not_applied_no_invented_patches" if large_text else "",
        }
        final_rows.append(final)
        clean_rows.append(
            {
                "dataset_group": "argentina/existing",
                "source_id": sid,
                "clip_id": clip_id,
                "existing_text": existing_text,
                "large_text": large_text,
                "turbo_text": turbo_text,
                "clean_text": "",
                "status": clean_status,
                "confidence": final["clean_confidence"],
                "patch_count": 0,
                "gpt_status": gpt_status,
                "notes": final["notes"],
            }
        )
    return final_rows, clean_rows


def build_new_discovery() -> list[dict[str, object]]:
    rows = []
    downloads = {row.get("url", ""): row for row in read_csv(LOCAL_DOWNLOADS) if row.get("status", "").startswith("downloaded_uploaded")}
    for row in read_csv(ARG_NEW):
        downloaded = downloads.get(row.get("url", ""), {})
        if downloaded:
            ingest_status = "source_downloaded_pending_clips_asr_roi"
            failure_reason = ""
            notes = "source video/audio downloaded locally and uploaded to GCS; clips/ASR/ROIs pending"
        else:
            ingest_status = "blocked_download_failed"
            failure_reason = "youtube_requires_login_or_cookies_from_vm"
            notes = "accepted source queued; VM yt-dlp metadata/download retry requires browser cookies or alternative source URL"
        rows.append(
            {
                "url": row.get("url", ""),
                "video_id": row.get("video_id", ""),
                "title": row.get("title", ""),
                "channel": row.get("channel", ""),
                "decision": row.get("decision", ""),
                "total_score": row.get("total_score", ""),
                "usable_minutes_estimate": row.get("usable_minutes_estimate", ""),
                "accepted_clips_estimate": row.get("accepted_clips_estimate", ""),
                "recommended_use": row.get("recommended_use", ""),
                "ingest_status": ingest_status,
                "failure_reason": failure_reason,
                "source_video_gcs_path": downloaded.get("gcs_video_path", ""),
                "source_audio_gcs_path": downloaded.get("gcs_audio_path", ""),
                "metadata_gcs_path": downloaded.get("gcs_info_path", ""),
                "notes": notes,
            }
        )
    return rows


def build_spanish_manifest() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(SPANISH):
        rows.append(
            {
                "dataset_group": "spanish_general/existing",
                "clip_id": row.get("clip_id", ""),
                "mp4_gcs_path": row.get("mp4_gcs_path", ""),
                "npz_gcs_path": row.get("npz_gcs_path", ""),
                "txt_gcs_path": row.get("txt_gcs_path", ""),
                "turbo_text_path": "",
                "asr_status": "blocked_missing_provenance_for_asr",
                "provenance_status": row.get("provenance_status", "not_documented_in_bucket"),
                "license_status": row.get("license_status", "unknown_or_sensitive"),
                "notes": "curriculum_visper has ROI/text assets but no reconstructable source URLs in current manifests",
            }
        )
    return rows


def append_failures(new_rows: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in read_csv(FAILURES)
        if not (
            row.get("stage") == "new_discovery_ingest"
            and row.get("dataset_group") == "argentina/new_discovery"
            and row.get("error_type") == "blocked_not_ingested_in_this_release"
        )
    ]
    seen = {(r.get("stage"), r.get("dataset_group"), r.get("source_id"), r.get("clip_id"), r.get("error_type")) for r in rows}
    for row in new_rows:
        key = (row.get("stage"), row.get("dataset_group"), row.get("source_id"), row.get("clip_id"), row.get("error_type"))
        if key not in seen:
            rows.append(row)
            seen.add(key)
    write_csv(FAILURES, rows, FAILURE_FIELDS)


def write_reports(final_rows: list[dict[str, object]], clean_rows: list[dict[str, object]], new_rows: list[dict[str, object]], spanish_rows: list[dict[str, object]]) -> None:
    clean_counts = Counter(str(r["clean_status"]) for r in final_rows)
    asr_counts = Counter(str(r["asr_status"]) for r in final_rows)
    align_counts = Counter(str(r["alignment_confidence"]) for r in final_rows)
    source_counts = Counter(str(r["source_id"]) for r in final_rows if r.get("source_id"))
    recon_rows = read_csv(RECON)
    recon_completed = [r for r in recon_rows if r.get("status") == "completed_reconstructed_audio"]
    asr_rows = [r for r in read_csv(ASR) if r.get("model_role") in {"large", "turbo"}]
    disagreement_rows = read_csv(DISAGREEMENT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FULL_REPORT.write_text(
        "\n".join(
            [
                "# Full clean release report",
                "",
                f"bucket: {DEST_BUCKET}/",
                "branch: feature/full-clean-release",
                "",
                "## Argentina existing",
                f"- total clips manifest: {len(final_rows)}",
                f"- sources: {len(source_counts)}",
                f"- clean_status_counts: {dict(clean_counts)}",
                f"- asr_status_counts: {dict(asr_counts)}",
                f"- alignment_confidence_counts: {dict(align_counts)}",
                f"- reconstructed_audio_clips: {len(recon_completed)}",
                f"- large_turbo_asr_rows: {len(asr_rows)}",
                f"- disagreement_rows: {len(disagreement_rows)}",
                "",
                "## Argentina new discovery",
                f"- accepted videos queued: {len(new_rows)}",
                f"- source_downloaded_pending_clips_asr_roi: {sum(1 for r in new_rows if r.get('ingest_status') == 'source_downloaded_pending_clips_asr_roi')}",
                f"- blocked_download_failed: {sum(1 for r in new_rows if r.get('ingest_status') == 'blocked_download_failed')}",
                "- reason for remaining blocked: yt-dlp on the VM requires YouTube login/cookies for accepted URLs; local download flow is now available.",
                "",
                "## Spanish general",
                f"- rows: {len(spanish_rows)}",
                "- ASR: blocked_missing_provenance_for_asr",
                "",
                "## GPT cleaning",
                f"- completed_clean_gpt: {sum(1 for r in clean_rows if r['status'] == 'completed_clean_gpt')}",
                f"- completed_large_turbo_no_gpt: {sum(1 for r in clean_rows if r['status'] == 'completed_large_turbo_no_gpt')}",
                f"- baseline_existing_only: {sum(1 for r in clean_rows if r['status'] == 'baseline_existing_only')}",
                "- no GPT patch was applied; no cleaning was invented.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    GPT_REPORT.write_text(
        "\n".join(
            [
                "# GPT cleaning report",
                "",
                "completed_clean_gpt: 0",
                f"completed_large_turbo_no_gpt: {sum(1 for r in clean_rows if r['status'] == 'completed_large_turbo_no_gpt')}",
                f"baseline_existing_only: {sum(1 for r in clean_rows if r['status'] == 'baseline_existing_only')}",
                f"needs_review_or_blocked: {sum(1 for r in clean_rows if r['status'] != 'completed_large_turbo_no_gpt')}",
                "",
                "No se aplicaron patches GPT en esta corrida. La regla fue no inventar limpieza sin salida JSONL validada.",
                "Los clips con ASR large/turbo usan `large_text` como selected_training_text.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    COST_REPORT.write_text(
        "\n".join(
            [
                "# Cost/runtime report",
                "",
                "project: labios-argentos-499900",
                "vm_runs:",
                "- name: vsr-full-clean-20260707-0200; zone: us-central1-a; machine_type: g2-standard-8; gpu: nvidia-l4; provisioning_model: SPOT; status: used_then_spot_terminated_then_deleted",
                "- name: vsr-full-clean-continue-20260707; zone: us-east1-d; machine_type: g2-standard-8; gpu: nvidia-l4; provisioning_model: STANDARD; status: used_for_resume_then_deleted",
                "outputs_synced_to_gcs: true",
                "cleanup_verification: data_release/reports/bucket_validation_report.md shows no matching instances/disks/static IPs",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    final_rows, clean_rows = build_existing_final()
    new_rows = build_new_discovery()
    spanish_rows = build_spanish_manifest()
    train_rows = [r for r in final_rows if r.get("split") == "train" and r.get("usable_for_training") == "true"]
    eval_rows = [r for r in final_rows if r.get("split") in {"val", "test"} and r.get("usable_for_eval") == "true"]
    write_csv(FINAL_RELEASE, final_rows, FINAL_FIELDS)
    write_csv(FINAL_TRAIN, train_rows, FINAL_FIELDS)
    write_csv(FINAL_EVAL, eval_rows, FINAL_FIELDS)
    write_csv(CLEAN_GPT, clean_rows, CLEAN_FIELDS)
    write_csv(NEW_DISCOVERY, new_rows, NEW_FIELDS)
    write_csv(SPANISH_REPORT, spanish_rows, SPANISH_FIELDS)
    append_failures(
        [
            {
                "stage": "vm_processing",
                "dataset_group": "argentina/existing",
                "source_id": "f09__ESTOY_EN_UN_BROTE_PERDON",
                "clip_id": "",
                "path": "gce://vsr-full-clean-20260707-0200",
                "error_type": "spot_vm_terminated",
                "error_message": "VM Spot terminated during reconstruction batch before the next source checkpoint completed.",
                "notes": "completed sources were synced to GCS; processing resumed on standard L4 VM",
            },
            {
                "stage": "new_discovery_ingest",
                "dataset_group": "argentina/new_discovery",
                "source_id": "",
                "clip_id": "",
                "path": str(ARG_NEW),
                "error_type": "blocked_download_failed",
                "error_message": "yt-dlp on the VM required YouTube login/cookies for accepted new_discovery URLs.",
                "notes": "accepted URLs remain queued in new_discovery_ingest_manifest.csv; no cookies were uploaded or logged",
            },
            {
                "stage": "spanish_general_asr",
                "dataset_group": "spanish_general/existing",
                "source_id": "",
                "clip_id": "",
                "path": "gs://labios-argentos-vsr-dataset/curriculum_visper/",
                "error_type": "blocked_missing_provenance_for_asr",
                "error_message": "No reconstructable source URLs/provenance found in current manifests.",
                "notes": "spanish_general kept separate; no GPT cleaning",
            },
        ]
    )
    write_reports(final_rows, clean_rows, new_rows, spanish_rows)
    print(f"final_release_rows={len(final_rows)} -> {FINAL_RELEASE}")
    print(f"train_rows={len(train_rows)} -> {FINAL_TRAIN}")
    print(f"eval_rows={len(eval_rows)} -> {FINAL_EVAL}")
    print(f"clean_rows={len(clean_rows)} -> {CLEAN_GPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
