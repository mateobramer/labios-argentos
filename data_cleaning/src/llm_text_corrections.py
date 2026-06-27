"""Validacion de correcciones textuales sugeridas por un LLM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


VALID_ACTIONS = {"keep", "corrected", "uncertain", "reject"}
REVIEW_RISK_FLAGS = {
    "nombre_propio",
    "cambio_grande",
    "posible_alucinacion",
    "requiere_video",
}


def cargar_clips_split(split_path: str | Path, source_id: str) -> list[dict[str, str]]:
    path = Path(split_path)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("titulo") == source_id:
                rows.append(row)
    rows.sort(key=lambda row: (_clip_number(row.get("clip", "")), row.get("clip", "")))
    return rows


def cargar_sugerencias(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "source_id" not in data or "corrections" not in data:
        raise ValueError("El JSON debe tener source_id y corrections")
    if not isinstance(data["corrections"], list):
        raise ValueError("corrections debe ser una lista")
    return data


def generar_revision(
    *,
    split_path: str | Path,
    suggestions_path: str | Path,
    output_dir: str | Path,
    clips_root: str | Path = "data/clips",
) -> dict[str, Any]:
    suggestions = cargar_sugerencias(suggestions_path)
    source_id = str(suggestions["source_id"])
    split_rows = cargar_clips_split(split_path, source_id)
    source_by_clip = {row["clip"]: row for row in split_rows}
    corrections = suggestions["corrections"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    suggestion_by_clip: dict[str, dict[str, Any]] = {}
    duplicates = []
    for item in corrections:
        clip_id = str(item.get("clip_id", ""))
        if clip_id in suggestion_by_clip:
            duplicates.append(clip_id)
        suggestion_by_clip[clip_id] = item

    manifest = []
    for row in split_rows:
        clip_id = row["clip"]
        source_raw = str(row.get("texto") or "")
        item = suggestion_by_clip.get(clip_id)
        if item is None:
            manifest.append(_missing_row(source_id, row, clips_root))
            continue
        manifest.append(_review_row(source_id, row, item, clips_root))

    known_clips = set(source_by_clip)
    unknown_clips = sorted(set(suggestion_by_clip) - known_clips)
    summary = _summary(manifest, duplicates, unknown_clips)

    raw_copy_path = output / "llm_suggestions.raw.json"
    if Path(suggestions_path).resolve() != raw_copy_path.resolve():
        shutil.copyfile(suggestions_path, raw_copy_path)
    _write_json(output / "summary.json", summary)
    _write_jsonl(output / "review_manifest.jsonl", manifest)
    _write_csv(output / "review_manifest.csv", manifest)
    return summary


def exportar_prompt_correccion(
    *,
    split_path: str | Path,
    source_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    rows = cargar_clips_split(split_path, source_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    continuous_text = " ".join(str(row.get("texto") or "").strip() for row in rows).strip()

    lines = [
        "# Correccion asistida clip por clip",
        "",
        f"Fuente: {source_id}",
        f"Split de referencia: {Path(split_path).as_posix()}",
        f"Cantidad de clips: {len(rows)}",
        "",
        "## Tarea",
        "",
        "Necesito corregir transcripciones de clips de un dataset de lectura de labios.",
        "",
        "Tenes una fuente completa dividida en clips consecutivos. Usa el contexto global "
        "para entender palabras raras, pero devolve una correccion clip por clip.",
        "",
        "## Reglas estrictas",
        "",
        "- No agregues informacion nueva.",
        "- No reescribas estilisticamente.",
        "- No mejores la redaccion si el texto ya se entiende.",
        "- Corregi solo errores evidentes de transcripcion.",
        "- Conserva muletillas, voseo, informalidad, insultos y repeticiones reales.",
        "- Si no estas seguro, no corrijas: marca action=\"uncertain\".",
        "- Si sospechas que podrias estar inventando, marca risk_flags.",
        "- No unas clips.",
        "- No cambies clip_id.",
        "- Devolve solo JSON valido, sin Markdown alrededor.",
        "",
        "## Formato de salida obligatorio",
        "",
        "{",
        f"  \"source_id\": \"{source_id}\",",
        "  \"corrections\": [",
        "    {",
        "      \"clip_id\": \"clip_0000\",",
        "      \"raw_text\": \"texto original exacto\",",
        "      \"suggested_text\": \"texto sugerido\",",
        "      \"action\": \"keep | corrected | uncertain\",",
        "      \"confidence\": 0.0,",
        "      \"risk_flags\": [],",
        "      \"notes\": \"\"",
        "    }",
        "  ]",
        "}",
        "",
        "## Reglas de consistencia",
        "",
        "- Si action=\"keep\", suggested_text debe ser igual a raw_text.",
        "- Si action=\"uncertain\", suggested_text debe ser igual a raw_text salvo que "
        "propongas una alternativa muy clara en notes.",
        "- Si action=\"corrected\", el cambio debe ser minimo y justificable por contexto.",
        "- confidence debe estar entre 0 y 1.",
        "- Tiene que haber exactamente una entrada por clip listado.",
        "",
        "## risk_flags posibles",
        "",
        "- \"nombre_propio\"",
        "- \"numero\"",
        "- \"jerga\"",
        "- \"insulto\"",
        "- \"muletilla\"",
        "- \"repeticion\"",
        "- \"cambio_grande\"",
        "- \"correccion_inferida\"",
        "- \"posible_alucinacion\"",
        "- \"requiere_video\"",
        "",
        "## Clips ordenados",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['clip']}: {str(row.get('texto') or '').strip()}")
    lines.extend(["", "## Texto continuo de contexto", "", continuous_text, ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(output), "clips": len(rows), "chars": len("\n".join(lines))}


def _review_row(
    source_id: str,
    split_row: dict[str, str],
    item: dict[str, Any],
    clips_root: str | Path,
) -> dict[str, Any]:
    clip_id = split_row["clip"]
    source_raw = str(split_row.get("texto") or "")
    llm_raw = str(item.get("raw_text") or "")
    suggested = str(item.get("suggested_text") or "")
    action = str(item.get("action") or "")
    confidence = _float_or_zero(item.get("confidence"))
    risk_flags = [str(flag) for flag in item.get("risk_flags", [])]

    validation_flags = []
    if action not in VALID_ACTIONS:
        validation_flags.append("accion_invalida")
    if _norm(source_raw) != _norm(llm_raw):
        validation_flags.append("raw_text_no_coincide")
    if not suggested.strip():
        validation_flags.append("suggested_text_vacio")
    if action == "keep" and _norm(suggested) != _norm(source_raw):
        validation_flags.append("keep_modifica_texto")
    if action == "corrected" and _norm(suggested) == _norm(source_raw):
        validation_flags.append("corrected_sin_cambio")
    if action == "uncertain" and _norm(suggested) != _norm(source_raw):
        validation_flags.append("uncertain_modifica_texto")
    if confidence < 0.7 and action == "corrected":
        validation_flags.append("correccion_baja_confianza")

    char_change_ratio = _change_ratio(source_raw, suggested)
    word_change_ratio = _word_change_ratio(source_raw, suggested)
    if char_change_ratio > 0.25 or word_change_ratio > 0.35:
        validation_flags.append("cambio_grande_detectado")

    review_status = _review_status(action, confidence, risk_flags, validation_flags)
    video_path = Path(clips_root) / source_id / f"{clip_id}.mp4"
    txt_path = Path(clips_root) / source_id / f"{clip_id}.txt"
    txt_file_text = _read_text_if_exists(txt_path)

    return {
        "source_id": source_id,
        "clip_id": clip_id,
        "split": split_row.get("split", ""),
        "speaker": split_row.get("spk", ""),
        "n_frames": split_row.get("n_frames", ""),
        "source_raw_text": source_raw,
        "llm_raw_text": llm_raw,
        "suggested_text": suggested,
        "action": action,
        "confidence": confidence,
        "char_change_ratio": round(char_change_ratio, 4),
        "word_change_ratio": round(word_change_ratio, 4),
        "risk_flags": risk_flags,
        "validation_flags": validation_flags,
        "review_status": review_status,
        "notes": str(item.get("notes") or ""),
        "video_path": video_path.as_posix(),
        "video_exists": video_path.exists(),
        "txt_path": txt_path.as_posix(),
        "txt_exists": txt_path.exists(),
        "txt_matches_source": _norm(txt_file_text) == _norm(source_raw),
    }


def _missing_row(source_id: str, split_row: dict[str, str], clips_root: str | Path) -> dict[str, Any]:
    clip_id = split_row["clip"]
    video_path = Path(clips_root) / source_id / f"{clip_id}.mp4"
    return {
        "source_id": source_id,
        "clip_id": clip_id,
        "split": split_row.get("split", ""),
        "speaker": split_row.get("spk", ""),
        "n_frames": split_row.get("n_frames", ""),
        "source_raw_text": str(split_row.get("texto") or ""),
        "llm_raw_text": "",
        "suggested_text": "",
        "action": "missing",
        "confidence": 0.0,
        "char_change_ratio": 0.0,
        "word_change_ratio": 0.0,
        "risk_flags": [],
        "validation_flags": ["correccion_faltante"],
        "review_status": "reject_suggestion",
        "notes": "",
        "video_path": video_path.as_posix(),
        "video_exists": video_path.exists(),
        "txt_path": (Path(clips_root) / source_id / f"{clip_id}.txt").as_posix(),
        "txt_exists": (Path(clips_root) / source_id / f"{clip_id}.txt").exists(),
        "txt_matches_source": _norm(_read_text_if_exists(Path(clips_root) / source_id / f"{clip_id}.txt"))
        == _norm(str(split_row.get("texto") or "")),
    }


def _summary(manifest: list[dict[str, Any]], duplicates: list[str], unknown_clips: list[str]) -> dict[str, Any]:
    actions = Counter(str(row["action"]) for row in manifest)
    review_status = Counter(str(row["review_status"]) for row in manifest)
    validation_flags = Counter(flag for row in manifest for flag in row["validation_flags"])
    risk_flags = Counter(flag for row in manifest for flag in row["risk_flags"])
    return {
        "clips": len(manifest),
        "actions": dict(sorted(actions.items())),
        "review_status": dict(sorted(review_status.items())),
        "validation_flags": dict(sorted(validation_flags.items())),
        "risk_flags": dict(sorted(risk_flags.items())),
        "duplicates": duplicates,
        "unknown_clips": unknown_clips,
        "video_exists": sum(1 for row in manifest if row["video_exists"]),
        "txt_exists": sum(1 for row in manifest if row["txt_exists"]),
        "txt_matches_source": sum(1 for row in manifest if row["txt_matches_source"]),
        "high_change": sum(
            1
            for row in manifest
            if row["char_change_ratio"] > 0.25 or row["word_change_ratio"] > 0.35
        ),
    }


def _review_status(
    action: str,
    confidence: float,
    risk_flags: list[str],
    validation_flags: list[str],
) -> str:
    if action == "keep":
        return "keep_raw"
    if action in {"uncertain", "reject", "missing"}:
        return "reject_suggestion"
    if validation_flags:
        return "reject_suggestion"
    if confidence < 0.75 and action == "corrected":
        return "reject_suggestion"
    if REVIEW_RISK_FLAGS.intersection(risk_flags):
        return "reject_suggestion"
    if action == "corrected":
        return "accept_correction"
    return "reject_suggestion"


def _write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            normalized["risk_flags"] = "|".join(row["risk_flags"])
            normalized["validation_flags"] = "|".join(row["validation_flags"])
            writer.writerow(normalized)


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _read_text_if_exists(path: str | Path) -> str:
    text_path = Path(path)
    if not text_path.exists():
        return ""
    return text_path.read_text(encoding="utf-8").strip()


def _change_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _word_change_ratio(a: str, b: str) -> float:
    aw = _norm(a).split()
    bw = _norm(b).split()
    if not aw and not bw:
        return 0.0
    return 1.0 - SequenceMatcher(None, aw, bw).ratio()


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip_number(clip_id: str) -> int:
    digits = "".join(ch for ch in clip_id if ch.isdigit())
    return int(digits) if digits else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida correcciones textuales sugeridas por LLM.")
    parser.add_argument("--export-prompt", action="store_true")
    parser.add_argument("--split", default="vsr_models/splits/val.csv")
    parser.add_argument("--source-id")
    parser.add_argument("--prompt-output")
    parser.add_argument("--suggestions")
    parser.add_argument("--output-dir")
    parser.add_argument("--clips-root", default="data/clips")
    args = parser.parse_args()

    if args.export_prompt:
        if not args.source_id or not args.prompt_output:
            parser.error("--export-prompt requiere --source-id y --prompt-output")
        result = exportar_prompt_correccion(
            split_path=args.split,
            source_id=args.source_id,
            output_path=args.prompt_output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if not args.suggestions or not args.output_dir:
        parser.error("validar requiere --suggestions y --output-dir")

    summary = generar_revision(
        split_path=args.split,
        suggestions_path=args.suggestions,
        output_dir=args.output_dir,
        clips_root=args.clips_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
