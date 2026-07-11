"""Valida el bucket limpio v1 con conteos y muestras chicas."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
PROJECT = "labios-argentos-499900"
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data_pipeline/release" / "reports" / "bucket_validation_report.md"

COUNT_PATTERNS = {
    "argentina_existing_mp4": f"{DEST_BUCKET}/argentina/existing/clips_mp4/**/*.mp4",
    "argentina_existing_npz": f"{DEST_BUCKET}/argentina/existing/rois_npz/**/*.npz",
    "argentina_existing_large_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/large/**/*.txt",
    "argentina_existing_clean_gpt_v1_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/clean_gpt_v1/**/*.txt",
    "argentina_existing_turbo_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/turbo/**/*.txt",
    "argentina_existing_large_reconstructed_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/large/**/*.txt",
    "argentina_existing_clips_with_audio": f"{DEST_BUCKET}/argentina/existing/clips_with_audio/**/*.mp4",
    "argentina_existing_reconstructed_audio": f"{DEST_BUCKET}/argentina/existing/reconstructed_audio/**/*.wav",
    "spanish_general_mp4": f"{DEST_BUCKET}/spanish_general/existing/clips_mp4/**/*.mp4",
    "spanish_general_npz": f"{DEST_BUCKET}/spanish_general/existing/rois_npz/**/*.npz",
    "spanish_general_large_txt": f"{DEST_BUCKET}/spanish_general/existing/transcripts/large/**/*.txt",
    "spanish_general_turbo_txt": f"{DEST_BUCKET}/spanish_general/existing/transcripts/turbo/**/*.txt",
    "context_packs": f"{DEST_BUCKET}/argentina/existing/metadata/context_packs/*.jsonl",
    "argentina_new_discovery_source_videos": f"{DEST_BUCKET}/argentina/new_discovery/source_videos/**/*",
    "argentina_new_discovery_source_audio": f"{DEST_BUCKET}/argentina/new_discovery/source_audio/**/*",
    "argentina_new_discovery_metadata": f"{DEST_BUCKET}/argentina/new_discovery/metadata/**/*",
    "argentina_new_discovery_clips_with_audio": f"{DEST_BUCKET}/argentina/new_discovery/clips_with_audio/**/*.mp4",
    "argentina_new_discovery_rois_npz": f"{DEST_BUCKET}/argentina/new_discovery/rois_npz/**/*.npz",
    "argentina_new_discovery_roi_mp4": f"{DEST_BUCKET}/argentina/new_discovery/clips_mp4/**/*.mp4",
    "argentina_new_discovery_large_txt": f"{DEST_BUCKET}/argentina/new_discovery/transcripts/large/**/*.txt",
    "argentina_new_discovery_turbo_txt": f"{DEST_BUCKET}/argentina/new_discovery/transcripts/turbo/**/*.txt",
}
RESOURCE_FILTER = "(name~vsr-full-clean OR name~vsr-cleaning-vm OR labels.task=full-clean-release)"


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


GSUTIL = tool("gsutil")
GCLOUD = tool("gcloud")


def run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def run_ok(args: list[str], timeout: int = 300) -> str:
    result = run(args, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Fallo comando: {' '.join(args)}\n{result.stderr[-2000:]}")
    return result.stdout


def gsutil_list(pattern: str, timeout: int = 600) -> list[str]:
    result = run([GSUTIL, "ls", pattern], timeout=timeout)
    if result.returncode != 0:
        if "matched no objects" in result.stderr:
            return []
        raise RuntimeError(f"Fallo gsutil ls {pattern}\n{result.stderr[-2000:]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("gs://")]


def count_manifest_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def count_optional_manifest_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return count_manifest_rows(path)


def copy_to_tmp(gcs_path: str, tmp_dir: Path) -> Path:
    local = tmp_dir / Path(gcs_path).name
    run_ok([GSUTIL, "cp", gcs_path, str(local)], timeout=180)
    return local


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - solo diagnostico
        raise RuntimeError("No se pudo resolver ffmpeg via imageio_ffmpeg") from exc


def probe_mp4(paths: list[str], tmp_dir: Path) -> list[dict[str, str]]:
    ffmpeg = ffmpeg_path()
    rows: list[dict[str, str]] = []
    for path in paths:
        local = copy_to_tmp(path, tmp_dir)
        result = run([ffmpeg, "-hide_banner", "-i", str(local)], timeout=60)
        combined = result.stdout + result.stderr
        rows.append(
            {
                "path": path,
                "has_video": str("Video:" in combined).lower(),
                "has_audio": str("Audio:" in combined).lower(),
                "audio_zero_marker": str("audio:0KiB" in combined).lower(),
            }
        )
    return rows


def probe_npz(paths: list[str], tmp_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        local = copy_to_tmp(path, tmp_dir)
        with np.load(local) as data:
            key = "rois" if "rois" in data.files else data.files[0]
            arr = data[key]
            rows.append(
                {
                    "path": path,
                    "key": key,
                    "shape": "x".join(str(x) for x in arr.shape),
                    "dtype": str(arr.dtype),
                }
            )
    return rows


def read_txt_samples(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        text = run_ok([GSUTIL, "cat", path], timeout=120).strip()
        rows.append({"path": path, "chars": str(len(text)), "sample": text[:120]})
    return rows


def main() -> None:
    counts = {name: len(gsutil_list(pattern, timeout=1200)) for name, pattern in COUNT_PATTERNS.items()}
    manifest_counts = {
        "argentina_existing_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "argentina_existing_manifest.csv"),
        "argentina_new_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "argentina_new_manifest.csv"),
        "spanish_general_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "spanish_general_manifest.csv"),
        "clean_manifest_rows": count_optional_manifest_rows(ROOT / "cleaning/gpt_clean_v1" / "outputs" / "clean_manifest.csv"),
        "alignment_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "alignment_manifest.csv"),
        "existing_reconstruction_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "existing_reconstruction_manifest.csv"),
        "asr_large_turbo_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "asr_large_turbo_manifest.csv"),
        "final_release_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "final_release_manifest.csv"),
        "final_train_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "final_train_manifest_clean_gpt_v1.csv"),
        "final_eval_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "final_eval_manifest_clean_gpt_v1.csv"),
        "new_discovery_ingest_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "new_discovery_ingest_manifest.csv"),
        "new_discovery_clip_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "new_discovery_clip_manifest.csv"),
        "new_discovery_asr_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "new_discovery_asr_manifest.csv"),
        "new_discovery_roi_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "new_discovery_roi_manifest.csv"),
        "spanish_general_asr_manifest_rows": count_manifest_rows(ROOT / "data_pipeline/release" / "spanish_general_asr_manifest.csv"),
    }

    with tempfile.TemporaryDirectory(prefix="clean_bucket_validation_") as tmp:
        tmp_dir = Path(tmp)
        mp4_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_mp4"], timeout=1200)[:5]
        audio_clip_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_clips_with_audio"], timeout=1200)[:5]
        npz_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_npz"], timeout=1200)[:5]
        txt_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_clean_gpt_v1_txt"], timeout=1200)[:5]
        large_txt_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_large_reconstructed_txt"], timeout=1200)[:5]
        turbo_txt_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_turbo_txt"], timeout=1200)[:5]
        new_source_video_samples = gsutil_list(COUNT_PATTERNS["argentina_new_discovery_source_videos"], timeout=1200)[:5]
        new_source_audio_samples = gsutil_list(COUNT_PATTERNS["argentina_new_discovery_source_audio"], timeout=1200)[:5]
        new_clip_samples = gsutil_list(COUNT_PATTERNS["argentina_new_discovery_clips_with_audio"], timeout=1200)[:5]
        new_npz_samples = gsutil_list(COUNT_PATTERNS["argentina_new_discovery_rois_npz"], timeout=1200)[:5]
        new_large_txt_samples = gsutil_list(COUNT_PATTERNS["argentina_new_discovery_large_txt"], timeout=1200)[:5]
        new_turbo_txt_samples = gsutil_list(COUNT_PATTERNS["argentina_new_discovery_turbo_txt"], timeout=1200)[:5]
        mp4_probe = probe_mp4(mp4_samples, tmp_dir)
        audio_clip_probe = probe_mp4(audio_clip_samples, tmp_dir)
        new_source_video_probe = probe_mp4(new_source_video_samples, tmp_dir)
        new_source_audio_probe = probe_mp4(new_source_audio_samples, tmp_dir)
        new_clip_probe = probe_mp4(new_clip_samples, tmp_dir)
        npz_probe = probe_npz(npz_samples, tmp_dir)
        new_npz_probe = probe_npz(new_npz_samples, tmp_dir)
        txt_probe = read_txt_samples(txt_samples)
        large_txt_probe = read_txt_samples(large_txt_samples)
        turbo_txt_probe = read_txt_samples(turbo_txt_samples)
        new_large_txt_probe = read_txt_samples(new_large_txt_samples)
        new_turbo_txt_probe = read_txt_samples(new_turbo_txt_samples)

    iam_policy = run_ok([GCLOUD, "storage", "buckets", "get-iam-policy", DEST_BUCKET], timeout=120)
    iam_ok = "user:fgutman@udesa.edu.ar" in iam_policy and "roles/storage.objectViewer" in iam_policy

    vm_list = run_ok(
        [GCLOUD, "compute", "instances", "list", "--project", PROJECT, "--filter", RESOURCE_FILTER, "--format", "value(name)"],
        timeout=120,
    ).strip()
    disk_list = run_ok(
        [GCLOUD, "compute", "disks", "list", "--project", PROJECT, "--filter", RESOURCE_FILTER, "--format", "value(name)"],
        timeout=120,
    ).strip()
    address_list = run_ok(
        [GCLOUD, "compute", "addresses", "list", "--project", PROJECT, "--filter", RESOURCE_FILTER, "--format", "value(name)"],
        timeout=120,
    ).strip()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bucket validation report",
        "",
        f"bucket: {DEST_BUCKET}/",
        f"iam_fg_object_viewer: {str(iam_ok).lower()}",
        f"resource_filter: {RESOURCE_FILTER}",
        f"remaining_vms_matching_filter: {json.dumps(vm_list)}",
        f"remaining_disks_matching_filter: {json.dumps(disk_list)}",
        f"remaining_static_addresses_matching_filter: {json.dumps(address_list)}",
        "",
        "## Counts",
        "",
    ]
    for name, value in counts.items():
        lines.append(f"- {name}: {value}")
    lines.extend(["", "## Manifest rows", ""])
    for name, value in manifest_counts.items():
        lines.append(f"- {name}: {value}")
    lines.extend(["", "## MP4 samples", ""])
    for row in mp4_probe:
        lines.append(f"- {row['path']} video={row['has_video']} audio={row['has_audio']} audio_zero_marker={row['audio_zero_marker']}")
    lines.extend(["", "## Reconstructed MP4 samples", ""])
    for row in audio_clip_probe:
        lines.append(f"- {row['path']} video={row['has_video']} audio={row['has_audio']} audio_zero_marker={row['audio_zero_marker']}")
    lines.extend(["", "## New discovery source samples", ""])
    for row in new_source_video_probe:
        lines.append(f"- source_video {row['path']} video={row['has_video']} audio={row['has_audio']} audio_zero_marker={row['audio_zero_marker']}")
    for row in new_source_audio_probe:
        lines.append(f"- source_audio {row['path']} video={row['has_video']} audio={row['has_audio']} audio_zero_marker={row['audio_zero_marker']}")
    lines.extend(["", "## New discovery clips_with_audio samples", ""])
    for row in new_clip_probe:
        lines.append(f"- {row['path']} video={row['has_video']} audio={row['has_audio']} audio_zero_marker={row['audio_zero_marker']}")
    lines.extend(["", "## NPZ samples", ""])
    for row in npz_probe:
        lines.append(f"- {row['path']} key={row['key']} shape={row['shape']} dtype={row['dtype']}")
    lines.extend(["", "## New discovery NPZ samples", ""])
    for row in new_npz_probe:
        lines.append(f"- {row['path']} key={row['key']} shape={row['shape']} dtype={row['dtype']}")
    lines.extend(["", "## clean_gpt_v1 TXT samples", ""])
    for row in txt_probe:
        lines.append(f"- {row['path']} chars={row['chars']} sample={row['sample']!r}")
    lines.extend(["", "## large/turbo TXT samples", ""])
    for row in large_txt_probe:
        lines.append(f"- large {row['path']} chars={row['chars']} sample={row['sample']!r}")
    for row in turbo_txt_probe:
        lines.append(f"- turbo {row['path']} chars={row['chars']} sample={row['sample']!r}")
    lines.extend(["", "## New discovery large/turbo TXT samples", ""])
    for row in new_large_txt_probe:
        lines.append(f"- large {row['path']} chars={row['chars']} sample={row['sample']!r}")
    for row in new_turbo_txt_probe:
        lines.append(f"- turbo {row['path']} chars={row['chars']} sample={row['sample']!r}")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"validation_report -> {REPORT}")


if __name__ == "__main__":
    main()
