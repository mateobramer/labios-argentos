"""Transcript cleaning restringido para experimentos VSR batch."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr_models" / "splits" / "splits.csv"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"

CHANGE_COLUMNS = [
    "source_id",
    "clip",
    "original_path",
    "cleaned_path",
    "original_text",
    "cleaned_text",
    "changed",
    "change_type",
    "evidence",
    "confidence",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir_texto(path: Path, texto: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto + "\n", encoding="utf-8")


def _quitar_controles_invalidos(texto: str) -> tuple[str, bool]:
    salida = []
    changed = False
    for ch in texto:
        if ch in "\t\n\r":
            salida.append(" ")
            changed = True
            continue
        if ch == "\ufffd" or unicodedata.category(ch).startswith("C"):
            changed = True
            continue
        salida.append(ch)
    return "".join(salida), changed


def limpiar_restringido(texto: str) -> tuple[str, list[str]]:
    cambios: list[str] = []
    normalizado = unicodedata.normalize("NFKC", texto)
    if normalizado != texto:
        cambios.append("unicode_normalization")

    sin_invalidos, invalidos = _quitar_controles_invalidos(normalizado)
    if invalidos:
        cambios.append("invalid_character_removed")

    espacios = re.sub(r"\s+", " ", sin_invalidos).strip()
    if espacios != sin_invalidos:
        cambios.append("space_normalization")

    return espacios, cambios


def source_txt_path(row: dict[str, str], repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data" / "clips" / row["titulo"] / f"{row['clip']}.txt"


def build_transcript_overlays(
    splits_path: Path = DEFAULT_SPLITS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    rows = leer_csv(splits_path)
    current_root = output_base / "transcripts_current"
    cleaned_root = output_base / "transcripts_cleaned_restricted"
    changes_csv = output_base / "transcript_cleaning_changes.csv"
    changes_csv.parent.mkdir(parents=True, exist_ok=True)

    change_rows: list[dict[str, str]] = []
    changed_count = 0

    for row in rows:
        source_id = row["titulo"]
        clip = row["clip"]
        original_text = row.get("texto", "")
        cleaned_text, changes = limpiar_restringido(original_text)

        current_path = current_root / source_id / f"{clip}.txt"
        cleaned_path = cleaned_root / source_id / f"{clip}.txt"
        escribir_texto(current_path, original_text.strip())
        escribir_texto(cleaned_path, cleaned_text)

        original_path = source_txt_path(row, repo_root)
        source_exists = original_path.exists()
        changed = cleaned_text != original_text
        changed_count += int(changed)
        change_type = ";".join(changes) if changes else "none"
        evidence = (
            f"split_csv_text; source_txt_exists={source_exists}; "
            "restricted_unicode_space_control_normalization"
            if changed
            else f"split_csv_text; source_txt_exists={source_exists}; sin cambios"
        )
        confidence = "high" if changed else "none"
        change_rows.append(
            {
                "source_id": source_id,
                "clip": clip,
                "original_path": str(original_path),
                "cleaned_path": str(cleaned_path),
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "changed": "true" if changed else "false",
                "change_type": change_type,
                "evidence": evidence,
                "confidence": confidence,
            }
        )

    with changes_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGE_COLUMNS)
        writer.writeheader()
        writer.writerows(change_rows)

    examples = [
        {
            "source_id": row["source_id"],
            "clip": row["clip"],
            "change_type": row["change_type"],
            "original_text": row["original_text"],
            "cleaned_text": row["cleaned_text"],
        }
        for row in change_rows
        if row["changed"] == "true"
    ][:10]

    return {
        "splits_path": str(splits_path),
        "transcripts": len(rows),
        "changed": changed_count,
        "unchanged": len(rows) - changed_count,
        "current_root": str(current_root),
        "cleaned_root": str(cleaned_root),
        "changes_csv": str(changes_csv),
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_transcript_overlays(args.splits, args.output_base)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
