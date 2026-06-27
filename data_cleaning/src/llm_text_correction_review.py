"""Helpers de revision para el notebook de correcciones LLM."""

from __future__ import annotations

import csv
import json
import subprocess
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import Audio, HTML, Markdown, Video, display

from data_cleaning.src.whisper_model_comparison import buscar_ffmpeg


def find_repo_root(start: str | Path | None = None) -> Path:
    root = Path(start or Path.cwd()).resolve()
    for candidate in [root, *root.parents]:
        if (candidate / "data_cleaning" / "src").exists():
            return candidate
    raise FileNotFoundError("No se encontro la raiz del repo")


def load_review_artifacts(root: Path, base_dir: str | Path) -> dict[str, Any]:
    base = root / base_dir
    summary_path = base / "summary.json"
    manifest_path = base / "review_manifest.csv"
    raw_suggestions_path = base / "llm_suggestions.raw.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "base_dir": base,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "raw_suggestions_path": raw_suggestions_path,
        "summary": summary,
        "rows": rows,
    }


def load_source_metadata(root: Path, source_id: str) -> dict[str, str]:
    path = root / "data" / "metadata" / "fuentes.csv"
    if not path.exists():
        return {"carpeta": source_id}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("carpeta") == source_id:
                return row
    return {"carpeta": source_id}


def summary_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in ["actions", "review_status", "validation_flags", "risk_flags"]:
        for key, value in summary.get(group, {}).items():
            rows.append({"grupo": group, "valor": key, "cantidad": value})
    rows.extend(
        [
            {"grupo": "coverage", "valor": "clips", "cantidad": summary["clips"]},
            {"grupo": "coverage", "valor": "video_exists", "cantidad": summary["video_exists"]},
            {"grupo": "coverage", "valor": "txt_exists", "cantidad": summary["txt_exists"]},
            {
                "grupo": "coverage",
                "valor": "txt_matches_source",
                "cantidad": summary.get("txt_matches_source", 0),
            },
            {"grupo": "coverage", "valor": "unknown_clips", "cantidad": len(summary["unknown_clips"])},
        ]
    )
    return rows


def accepted_correction_rows(rows: list[dict[str, str]], limit: int = 12) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["review_status"] == "accept_correction"]
    return [
        {
            "clip": row["clip_id"],
            "raw": trunc(row["source_raw_text"], 90),
            "suggested": trunc(row["suggested_text"], 90),
            "confidence": row["confidence"],
            "risk_flags": row["risk_flags"] or "-",
        }
        for row in candidates[:limit]
    ]


def rejected_suggestion_rows(rows: list[dict[str, str]], limit: int = 15) -> list[dict[str, Any]]:
    rejected = [row for row in rows if row["review_status"] == "reject_suggestion"]
    return [
        {
            "clip": row["clip_id"],
            "action": row["action"],
            "confidence": row["confidence"],
            "raw_repo": trunc(row["source_raw_text"], 75),
            "raw_llm": trunc(row["llm_raw_text"], 75),
            "suggested": trunc(row["suggested_text"], 75),
            "validation": row["validation_flags"] or "-",
            "risk": row["risk_flags"] or "-",
        }
        for row in rejected[:limit]
    ]


def select_diff_examples(rows: list[dict[str, str]], per_group: int = 4) -> list[dict[str, str]]:
    accepted = [row for row in rows if row["review_status"] == "accept_correction"]
    rejected = [row for row in rows if row["review_status"] == "reject_suggestion"]
    return accepted[:per_group] + rejected[:per_group]


def render_diff_examples(rows: list[dict[str, str]]) -> None:
    table_rows = replacement_table_rows(rows)
    show_table(
        table_rows,
        ["clip", "decision", "action", "confidence", "antes", "despues", "risk", "validation"],
    )


def replacement_table_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        replacements = word_replacements(row["source_raw_text"], row["suggested_text"])
        if not replacements:
            replacements = [("-", "-")]
        for before, after in replacements:
            table.append(
                {
                    "clip": row["clip_id"],
                    "decision": row["review_status"],
                    "action": row["action"],
                    "confidence": row["confidence"],
                    "antes": before,
                    "despues": after,
                    "risk": row["risk_flags"] or "-",
                    "validation": row["validation_flags"] or "-",
                }
            )
    return table


def word_replacements(raw: str, suggested: str) -> list[tuple[str, str]]:
    raw_words = str(raw).split()
    sug_words = str(suggested).split()
    matcher = SequenceMatcher(None, raw_words, sug_words)
    replacements = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before = " ".join(raw_words[i1:i2]) or "-"
        after = " ".join(sug_words[j1:j2]) or "-"
        replacements.append((before, after))
    return replacements


def select_video_examples(rows: list[dict[str, str]], limit: int = 4) -> list[dict[str, str]]:
    candidates = [row for row in rows if row["review_status"] == "accept_correction"]
    rejected = [row for row in rows if row["review_status"] == "reject_suggestion"]
    selected = []
    for bucket in [candidates[:2], rejected[:2], candidates[2:], rejected[2:]]:
        for row in bucket:
            if row["clip_id"] in {item["clip_id"] for item in selected}:
                continue
            selected.append(row)
            if len(selected) >= limit:
                return selected
    return selected


def export_review_webm(
    root: Path,
    rows: list[dict[str, str]],
    output_dir: str | Path,
    *,
    force: bool = False,
) -> list[dict[str, str]]:
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = buscar_ffmpeg()
    exported = []
    for row in rows:
        source_path = root / row["video_path"]
        target_path = output / f"{row['clip_id']}.webm"
        vscode_mp4_path = output / f"{row['clip_id']}_vscode.mp4"
        audio_path = output / f"{row['clip_id']}.mp3"
        thumb_path = output / f"{row['clip_id']}.jpg"
        if force or not target_path.exists():
            cmd = [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-c:v",
                "libvpx",
                "-crf",
                "18",
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "libvorbis",
                "-q:a",
                "5",
                str(target_path),
            ]
            subprocess.run(cmd, check=True)
        if force or not vscode_mp4_path.exists():
            # VS Code webviews reproducen H.264, pero no AAC; usamos MP3 para el audio.
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(vscode_mp4_path),
                ],
                check=True,
            )
        if force or not audio_path.exists():
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    str(audio_path),
                ],
                check=True,
            )
        if force or not thumb_path.exists():
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    "0.7",
                    "-i",
                    str(source_path),
                    "-frames:v",
                    "1",
                    str(thumb_path),
                ],
                check=True,
            )
        copied = dict(row)
        copied["review_webm_path"] = target_path.resolve().as_posix()
        copied["review_webm_exists"] = target_path.exists()
        copied["review_mp4_path"] = vscode_mp4_path.resolve().as_posix()
        copied["review_mp4_exists"] = vscode_mp4_path.exists()
        copied["review_audio_path"] = audio_path.resolve().as_posix()
        copied["review_audio_exists"] = audio_path.exists()
        copied["review_thumbnail_path"] = thumb_path.resolve().as_posix()
        copied["review_thumbnail_exists"] = thumb_path.exists()
        exported.append(copied)
    return exported


def display_review_video(row: dict[str, str], root: Path, width: int = 640) -> None:
    """Muestra video con audio compatible con VS Code y deja links de respaldo."""
    source_mp4 = (root / row["video_path"]).resolve()
    review_mp4 = Path(row.get("review_mp4_path") or source_mp4).resolve()
    if not review_mp4.exists():
        display(Markdown(f"_No se encontro el video: `{review_mp4}`_"))
        return

    display(
        Video(
            filename=str(review_mp4),
            embed=True,
            mimetype="video/mp4",
            html_attributes=(
                "controls preload='metadata' playsinline "
                f"style='width:{width}px;max-width:100%;height:auto;background:#111;'"
            ),
        )
    )

    audio_path = row.get("review_audio_path")
    if audio_path:
        audio = Path(audio_path).resolve()
        if audio.exists():
            display(Markdown("Audio separado de respaldo:"))
            display(Audio(filename=str(audio), embed=True))

    links = [
        f"<a href='{review_mp4.as_uri()}'>Abrir MP4 compatible VS Code</a>",
        f"<a href='{source_mp4.as_uri()}'>Abrir MP4 original</a>",
    ]
    if audio_path:
        audio = Path(audio_path).resolve()
        if audio.exists():
            links.append(f"<a href='{audio.as_uri()}'>Abrir MP3</a>")
    webm_path = row.get("review_webm_path")
    if webm_path:
        webm = Path(webm_path).resolve()
        if webm.exists():
            links.append(f"<a href='{webm.as_uri()}'>Abrir WebM generado</a>")
    display(HTML("<p style='font-size:12px'>" + " | ".join(links) + "</p>"))


def show_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    max_chars: int = 90,
    max_width_px: int = 1100,
) -> None:
    if not rows:
        display(Markdown("_Sin filas para mostrar._"))
        return

    header = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{_fmt_cell(row.get(col, ''), max_chars=max_chars)}</td>" for col in columns
        )
        body.append(f"<tr>{cells}</tr>")

    html = f"""
    <div style="max-width:{max_width_px}px;overflow-x:auto;margin:0.4rem 0 0.8rem 0;">
      <table style="
          border-collapse:collapse;
          width:100%;
          table-layout:fixed;
          font-size:12px;
          line-height:1.25;
      ">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    <style>
      table th, table td {{
        border: 1px solid #ddd;
        padding: 4px 6px;
        vertical-align: top;
        overflow-wrap: anywhere;
        word-break: normal;
      }}
      table th {{
        background: rgba(127, 127, 127, 0.12);
        font-weight: 600;
      }}
    </style>
    """
    display(HTML(html))


def display_source_metadata(metadata: dict[str, str]) -> None:
    show_table(
        [
            {
                "carpeta": metadata.get("carpeta", ""),
                "hablante_principal": metadata.get("hablante_principal", ""),
                "canal_o_fuente": metadata.get("canal_o_fuente", ""),
                "url": metadata.get("url", ""),
                "notas": trunc(metadata.get("notas", ""), 120),
            }
        ],
        ["carpeta", "hablante_principal", "canal_o_fuente", "url", "notas"],
    )


def trunc(value: Any, n: int = 120) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def _fmt_cell(value: Any, *, max_chars: int) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = trunc(value, max_chars)
    return escape(text)
