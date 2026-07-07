"""Descarga fuentes localmente y sube solo artefactos sanitizados a GCS.

El flujo intenta primero `yt-dlp` normal. Si YouTube pide login, reintenta con
`--cookies-from-browser` usando cookies locales del navegador, sin exportarlas a
archivo ni escribirlas en logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_release"
DISCOVERY = ROOT / "data_discovery" / "outputs"
PACK_DIR = OUT_DIR / "human_review_pack"
LOCAL_DIR = OUT_DIR / "local_sources"
REPORT = OUT_DIR / "reports" / "local_source_download_report.md"
FAILURES = OUT_DIR / "reports" / "failures.csv"
DOWNLOAD_MANIFEST = OUT_DIR / "local_source_download_manifest.csv"

DEST_BUCKET = "gs://labios-argentos-vsr-clean-v1"
GCS_NEW_VIDEO = f"{DEST_BUCKET}/argentina/new_discovery/source_videos"
GCS_NEW_AUDIO = f"{DEST_BUCKET}/argentina/new_discovery/source_audio"
GCS_NEW_META = f"{DEST_BUCKET}/argentina/new_discovery/metadata"
GCS_EXIST_VIDEO = f"{DEST_BUCKET}/argentina/existing/source_videos"
GCS_EXIST_AUDIO = f"{DEST_BUCKET}/argentina/existing/source_audio"
GCS_EXIST_META = f"{DEST_BUCKET}/argentina/existing/metadata/source_downloads"

FAILURE_FIELDS = ["stage", "dataset_group", "source_id", "clip_id", "path", "error_type", "error_message", "notes"]
DOWNLOAD_FIELDS = [
    "dataset_group",
    "source_id",
    "url",
    "status",
    "used_cookies_from_browser",
    "local_video_path",
    "local_audio_path",
    "gcs_video_path",
    "gcs_audio_path",
    "gcs_info_path",
    "gcs_log_path",
    "notes",
]


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
    if not found:
        raise RuntimeError(f"No se encontro {name}")
    return found


def ytdlp_cmd() -> list[str]:
    found = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if found:
        return [found]
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("No se encontro yt-dlp ni modulo yt_dlp") from exc
    return [str(Path(shutil.which("python") or "python")), "-m", "yt_dlp"]


def run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)


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


def append_failure(row: dict[str, object]) -> None:
    rows = read_csv(FAILURES)
    key = (row.get("stage"), row.get("dataset_group"), row.get("source_id"), row.get("path"), row.get("error_type"))
    seen = {(r.get("stage"), r.get("dataset_group"), r.get("source_id"), r.get("path"), r.get("error_type")) for r in rows}
    if key not in seen:
        rows.append(row)
    write_csv(FAILURES, rows, FAILURE_FIELDS)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "source").strip("_")[:120]


def needs_cookies(stderr: str) -> bool:
    lowered = stderr.casefold()
    return "sign in to confirm" in lowered or "cookies" in lowered or "not a bot" in lowered or "login" in lowered


def sanitize_log(text: str) -> str:
    text = re.sub(r"(?i)(cookie|authorization|x-goog-[^:]+):[^\n\r]+", r"\1:<redacted>", text)
    text = re.sub(r"(?i)(--cookies(?:-from-browser)?)(\s+)(\S+)", r"\1\2<redacted>", text)
    return text[-12000:]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def upload_file(local: Path, gcs_prefix: str) -> str:
    gcloud = tool("gcloud")
    dest = f"{gcs_prefix}/{local.name}"
    result = run([gcloud, "storage", "cp", str(local), dest], timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:])
    return dest


def split_media_candidates(work: Path) -> tuple[Path | None, Path | None]:
    videos = [p for p in work.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv"} and ".f251" not in p.name]
    audios = [p for p in work.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".m4a", ".webm", ".opus"} and (".f251" in p.name or p.suffix.lower() != ".mp4")]
    video = max(videos, key=lambda p: p.stat().st_size) if videos else None
    audio = max(audios, key=lambda p: p.stat().st_size) if audios else None
    return video, audio


def download_one(row: dict[str, str], dataset: str, browser: str, audio_only: bool) -> dict[str, object]:
    url = row.get("url") or row.get("manual_alternative_url") or row.get("current_candidate_url") or row.get("manual_url")
    source_id = row.get("source_id") or row.get("video_id") or safe_id(url)
    video_id = row.get("video_id") or source_id
    work = LOCAL_DIR / dataset / safe_id(source_id)
    work.mkdir(parents=True, exist_ok=True)
    base = work / safe_id(video_id)
    info_path = work / f"{safe_id(video_id)}.info.json"
    log_path = work / f"{safe_id(video_id)}.download.log"

    common = ytdlp_cmd() + [
        "--no-playlist",
        "--no-part",
        "--write-info-json",
        "--write-auto-subs",
        "--sub-langs",
        "es,es-orig",
        "--convert-subs",
        "vtt",
        "--merge-output-format",
        "mp4",
        "-o",
        str(base) + ".%(ext)s",
    ]
    if audio_only:
        common += ["-f", "ba/b", "-x", "--audio-format", "wav"]
    else:
        common += ["-f", "bv*[height<=720]+ba/b[height<=720]/b"]

    first = run(common + [url], timeout=3600)
    used_cookies = False
    result = first
    if first.returncode != 0 and needs_cookies(first.stderr + first.stdout):
        used_cookies = True
        result = run(common + ["--cookies-from-browser", browser, url], timeout=3600)

    write_text(log_path, sanitize_log((result.stdout or "") + "\n" + (result.stderr or "")))
    video_after_error, audio_after_error = split_media_candidates(work)
    recoverable_existing_media = bool(video_after_error or audio_after_error) and not needs_cookies(result.stderr + result.stdout)
    if result.returncode != 0 and not recoverable_existing_media:
        append_failure(
            {
                "stage": "local_source_download",
                "dataset_group": dataset.replace("_", "/"),
                "source_id": source_id,
                "clip_id": "",
                "path": url,
                "error_type": "blocked_download_failed",
                "error_message": sanitize_log(result.stderr or result.stdout)[-1000:],
                "notes": f"used_cookies_from_browser={str(used_cookies).lower()}",
            }
        )
        return {
            "dataset_group": dataset.replace("_", "/"),
            "source_id": source_id,
            "url": url,
            "status": "blocked_download_failed",
            "used_cookies_from_browser": str(used_cookies).lower(),
            "local_media_path": "",
            "gcs_media_path": "",
            "gcs_info_path": "",
            "log_path": str(log_path),
        }

    video, audio = split_media_candidates(work)
    if not video and not audio:
        raise RuntimeError(f"yt-dlp no genero media en {work}")
    if not info_path.exists():
        infos = sorted(work.glob("*.info.json"))
        if infos:
            info_path = infos[0]

    if dataset == "argentina_new_discovery":
        meta_prefix = f"{GCS_NEW_META}/{safe_id(source_id)}"
        video_prefix = f"{GCS_NEW_VIDEO}/{safe_id(source_id)}"
        audio_prefix = f"{GCS_NEW_AUDIO}/{safe_id(source_id)}"
    else:
        meta_prefix = f"{GCS_EXIST_META}/{safe_id(source_id)}"
        video_prefix = f"{GCS_EXIST_VIDEO}/{safe_id(source_id)}"
        audio_prefix = f"{GCS_EXIST_AUDIO}/{safe_id(source_id)}"

    gcs_video = upload_file(video, video_prefix) if video else ""
    gcs_audio = upload_file(audio, audio_prefix) if audio else ""
    gcs_info = upload_file(info_path, meta_prefix) if info_path.exists() else ""
    gcs_log = upload_file(log_path, meta_prefix)

    return {
        "dataset_group": dataset.replace("_", "/"),
        "source_id": source_id,
        "url": url,
        "status": "downloaded_uploaded" if result.returncode == 0 else "downloaded_uploaded_partial_ytdlp_error",
        "used_cookies_from_browser": str(used_cookies).lower(),
        "local_video_path": str(video or ""),
        "local_audio_path": str(audio or ""),
        "gcs_video_path": gcs_video,
        "gcs_audio_path": gcs_audio,
        "gcs_info_path": gcs_info,
        "gcs_log_path": gcs_log,
        "notes": "" if result.returncode == 0 else sanitize_log(result.stderr or result.stdout)[-500:],
    }


def selected_rows(args: argparse.Namespace) -> tuple[str, list[dict[str, str]]]:
    if args.new_discovery:
        rows = read_csv(PACK_DIR / "human_new_discovery_download_needed.csv")
        dataset = "argentina_new_discovery"
        rows = [r for r in rows if not args.only_manual or r.get("manual_alternative_url") or r.get("manual_download_path")]
    elif args.existing:
        rows = read_csv(PACK_DIR / "human_source_mapping_needed.csv")
        dataset = "argentina_existing"
        rows = [r for r in rows if r.get("manual_url") or (not args.only_manual and r.get("current_candidate_url"))]
        for row in rows:
            row["url"] = row.get("manual_url") or row.get("current_candidate_url", "")
    else:
        raise SystemExit("Usar --new-discovery o --existing")
    if args.limit is not None:
        rows = rows[: args.limit]
    return dataset, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-discovery", action="store_true")
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--browser", default="chrome")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--only-manual", action="store_true", help="Usar solo filas con overrides manuales.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset, rows = selected_rows(args)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        try:
            result = download_one(row, dataset, args.browser, args.audio_only)
        except Exception as exc:  # noqa: BLE001
            result = {
                "dataset_group": dataset.replace("_", "/"),
                "source_id": row.get("source_id") or row.get("video_id") or "",
                "url": row.get("url") or row.get("current_candidate_url") or "",
                "status": "blocked_download_failed",
                "used_cookies_from_browser": "",
                "local_video_path": "",
                "local_audio_path": "",
                "gcs_video_path": "",
                "gcs_audio_path": "",
                "gcs_info_path": "",
                "gcs_log_path": "",
                "notes": str(exc)[-500:],
            }
        results.append(result)
        print(f"{result.get('status')} {result.get('source_id')} video={result.get('gcs_video_path', '')} audio={result.get('gcs_audio_path', '')}")

    lines = ["# Local source download report", "", f"dataset: {dataset}", f"rows_attempted: {len(results)}", ""]
    for result in results:
        lines.append(
            f"- {result.get('source_id')}: {result.get('status')} cookies={result.get('used_cookies_from_browser')} video={result.get('gcs_video_path','')} audio={result.get('gcs_audio_path','')}"
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    existing = read_csv(DOWNLOAD_MANIFEST)
    merged = {(row.get("dataset_group"), row.get("source_id"), row.get("url")): row for row in existing}
    for result in results:
        merged[(result.get("dataset_group"), result.get("source_id"), result.get("url"))] = result
    write_csv(DOWNLOAD_MANIFEST, merged.values(), DOWNLOAD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
