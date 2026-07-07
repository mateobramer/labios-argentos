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
REPORT = ROOT / "data_release" / "reports" / "bucket_validation_report.md"

COUNT_PATTERNS = {
    "argentina_existing_mp4": f"{DEST_BUCKET}/argentina/existing/clips_mp4/**/*.mp4",
    "argentina_existing_npz": f"{DEST_BUCKET}/argentina/existing/rois_npz/**/*.npz",
    "argentina_existing_large_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/large/**/*.txt",
    "argentina_existing_clean_v1_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/clean_v1/**/*.txt",
    "argentina_existing_turbo_txt": f"{DEST_BUCKET}/argentina/existing/transcripts/turbo/**/*.txt",
    "spanish_general_mp4": f"{DEST_BUCKET}/spanish_general/existing/clips_mp4/**/*.mp4",
    "spanish_general_npz": f"{DEST_BUCKET}/spanish_general/existing/rois_npz/**/*.npz",
    "spanish_general_large_txt": f"{DEST_BUCKET}/spanish_general/existing/transcripts/large/**/*.txt",
    "spanish_general_turbo_txt": f"{DEST_BUCKET}/spanish_general/existing/transcripts/turbo/**/*.txt",
    "context_packs": f"{DEST_BUCKET}/argentina/existing/metadata/context_packs/*.jsonl",
}


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
        result = run([ffmpeg, "-hide_banner", "-i", str(local), "-f", "null", "-"], timeout=120)
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
        "argentina_existing_manifest_rows": count_manifest_rows(ROOT / "data_release" / "argentina_existing_manifest.csv"),
        "argentina_new_manifest_rows": count_manifest_rows(ROOT / "data_release" / "argentina_new_manifest.csv"),
        "spanish_general_manifest_rows": count_manifest_rows(ROOT / "data_release" / "spanish_general_manifest.csv"),
        "clean_manifest_rows": count_manifest_rows(ROOT / "data_cleaning_clean_v1" / "outputs" / "clean_manifest.csv"),
    }

    with tempfile.TemporaryDirectory(prefix="clean_bucket_validation_") as tmp:
        tmp_dir = Path(tmp)
        mp4_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_mp4"], timeout=1200)[:5]
        npz_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_npz"], timeout=1200)[:5]
        txt_samples = gsutil_list(COUNT_PATTERNS["argentina_existing_clean_v1_txt"], timeout=1200)[:5]
        mp4_probe = probe_mp4(mp4_samples, tmp_dir)
        npz_probe = probe_npz(npz_samples, tmp_dir)
        txt_probe = read_txt_samples(txt_samples)

    iam_policy = run_ok([GCLOUD, "storage", "buckets", "get-iam-policy", DEST_BUCKET], timeout=120)
    iam_ok = "user:fgutman@udesa.edu.ar" in iam_policy and "roles/storage.objectViewer" in iam_policy

    vm_list = run_ok(
        [GCLOUD, "compute", "instances", "list", "--project", PROJECT, "--filter", "name~vsr-cleaning-vm", "--format", "value(name)"],
        timeout=120,
    ).strip()
    disk_list = run_ok(
        [GCLOUD, "compute", "disks", "list", "--project", PROJECT, "--filter", "name~vsr-cleaning-vm", "--format", "value(name)"],
        timeout=120,
    ).strip()
    address_list = run_ok(
        [GCLOUD, "compute", "addresses", "list", "--project", PROJECT, "--filter", "name~vsr-cleaning-vm", "--format", "value(name)"],
        timeout=120,
    ).strip()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bucket validation report",
        "",
        f"bucket: {DEST_BUCKET}/",
        f"iam_fg_object_viewer: {str(iam_ok).lower()}",
        f"remaining_vms_named_vsr_cleaning: {json.dumps(vm_list)}",
        f"remaining_disks_named_vsr_cleaning: {json.dumps(disk_list)}",
        f"remaining_static_addresses_named_vsr_cleaning: {json.dumps(address_list)}",
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
    lines.extend(["", "## NPZ samples", ""])
    for row in npz_probe:
        lines.append(f"- {row['path']} key={row['key']} shape={row['shape']} dtype={row['dtype']}")
    lines.extend(["", "## clean_v1 TXT samples", ""])
    for row in txt_probe:
        lines.append(f"- {row['path']} chars={row['chars']} sample={row['sample']!r}")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"validation_report -> {REPORT}")


if __name__ == "__main__":
    main()
