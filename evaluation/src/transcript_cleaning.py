"""Transcript cleaning seguro + candidatos agresivos revisables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr_models" / "splits" / "splits.csv"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
DEFAULT_LEXICON = REPO_ROOT / "evaluation" / "experiments" / "batch_vsr" / "entity_lexicon.csv"
VISUAL_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "visual_cleaning"

NON_SPEECH_MARKERS = {
    "musica",
    "música",
    "aplausos",
    "risas",
    "subtitulos",
    "subtítulos",
    "suscribete",
    "suscríbete",
}

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
    "auto_applied",
]

CANDIDATE_COLUMNS = [
    "source_id",
    "clip",
    "original_text",
    "candidate_type",
    "span",
    "suggestion",
    "evidence",
    "confidence",
    "auto_applied",
    "reason_not_auto_applied",
]

QUALITY_COLUMNS = [
    "source_id",
    "clip",
    "split",
    "transcript_usability",
    "transcript_reasons",
    "transcript_policy_moderate",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def escribir_texto(path: Path, texto: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto + "\n", encoding="utf-8")


def source_txt_path(row: dict[str, str], repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data" / "clips" / row["titulo"] / f"{row['clip']}.txt"


def normalizar_token(token: str) -> str:
    token = unicodedata.normalize("NFKD", token.lower())
    return "".join(ch for ch in token if not unicodedata.combining(ch))


def cargar_lexicon(path: Path = DEFAULT_LEXICON) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return leer_csv(path)


def _entity_aliases(entry: dict[str, str]) -> list[str]:
    aliases = entry.get("aliases", "")
    return [a.strip() for a in re.split(r"[;|]", aliases) if a.strip()]


def _entity_hints(entry: dict[str, str]) -> list[str]:
    hints = entry.get("source_hint", "")
    return [normalizar_token(h.strip()) for h in re.split(r"[;|]", hints) if h.strip()]


def _distancia_chica(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 2:
        return False
    ratio = SequenceMatcher(None, normalizar_token(a), normalizar_token(b)).ratio()
    return ratio >= 0.78


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


def _remover_markers_no_hablados(texto: str) -> tuple[str, list[str]]:
    removidos: list[str] = []
    patron = re.compile(
        r"(?i)(\[(m[uú]sica|aplausos|risas|subt[ií]tulos|suscr[ií]bete)\]|"
        r"\((m[uú]sica|aplausos|risas|subt[ií]tulos|suscr[ií]bete)\)|"
        r"\b(subt[ií]tulos|suscr[ií]bete)\b)"
    )

    def reemplazar(match: re.Match[str]) -> str:
        removidos.append(match.group(0))
        return " "

    return patron.sub(reemplazar, texto), removidos


def _aplicar_entidades(texto: str, source_id: str, lexicon: list[dict[str, str]]) -> tuple[str, list[str]]:
    cambios: list[str] = []
    if not lexicon:
        return texto, cambios
    source_norm = normalizar_token(source_id)
    tokens = texto.split()
    changed = False
    for i, token in enumerate(tokens):
        clean_token = token.strip(".,;:!?¿¡()[]")
        for entry in lexicon:
            canonical = entry.get("canonical", "").strip()
            if not canonical:
                continue
            hints = _entity_hints(entry)
            if hints and not any(hint in source_norm for hint in hints):
                continue
            for alias in _entity_aliases(entry):
                if normalizar_token(clean_token) == normalizar_token(alias) and _distancia_chica(clean_token, canonical):
                    tokens[i] = token.replace(clean_token, canonical)
                    cambios.append(f"{clean_token}->{canonical}")
                    changed = True
    return (" ".join(tokens) if changed else texto), cambios


def auto_clean_safe(texto: str, source_id: str = "", lexicon: list[dict[str, str]] | None = None) -> tuple[str, list[str], list[str]]:
    lexicon = lexicon or []
    cambios: list[str] = []
    evidence: list[str] = []

    normalizado = unicodedata.normalize("NFKC", texto)
    if normalizado != texto:
        cambios.append("unicode_normalization")
        evidence.append("NFKC")

    sin_invalidos, invalidos = _quitar_controles_invalidos(normalizado)
    if invalidos:
        cambios.append("invalid_character_removed")
        evidence.append("caracteres invisibles/control removidos")

    sin_markers, markers = _remover_markers_no_hablados(sin_invalidos)
    if markers:
        cambios.append("non_speech_marker_removed")
        evidence.append("markers completos removidos: " + "|".join(markers[:5]))

    con_entidades, entity_changes = _aplicar_entidades(sin_markers, source_id, lexicon)
    if entity_changes:
        cambios.append("entity_restricted")
        evidence.append("lexicon+source_hint: " + "|".join(entity_changes[:5]))

    espacios = re.sub(r"\s+", " ", con_entidades).strip()
    if espacios != con_entidades:
        cambios.append("space_normalization")
        evidence.append("espacios multiples/strip")

    return espacios, cambios, evidence


def limpiar_restringido(texto: str) -> tuple[str, list[str]]:
    cleaned, changes, _ = auto_clean_safe(texto)
    return cleaned, changes


def _tokens(texto: str) -> list[str]:
    return [t for t in re.split(r"\s+", texto.strip()) if t]


def _garbage_ratio(texto: str) -> float:
    if not texto:
        return 0.0
    valid = 0
    for ch in texto:
        if ch.isalnum() or ch.isspace() or ch in "áéíóúüñÁÉÍÓÚÜÑ-_'":
            valid += 1
    return 1.0 - (valid / len(texto))


def detectar_candidates(row: dict[str, str], cleaned_text: str) -> list[dict[str, str]]:
    source_id = row["titulo"]
    clip = row["clip"]
    original_text = row.get("texto", "")
    tokens = _tokens(cleaned_text)
    candidates: list[dict[str, str]] = []

    def add(candidate_type: str, span: str, suggestion: str, evidence: str, confidence: str = "medium") -> None:
        candidates.append(
            {
                "source_id": source_id,
                "clip": clip,
                "original_text": original_text,
                "candidate_type": candidate_type,
                "span": span,
                "suggestion": suggestion,
                "evidence": evidence,
                "confidence": confidence,
                "auto_applied": "false",
                "reason_not_auto_applied": "requiere revision humana; no hay evidencia fuerte para reescribir",
            }
        )

    if not cleaned_text:
        add("empty_text", "", "", "texto vacio despues de limpieza", "high")
        return candidates

    if not any(ch.isalpha() for ch in cleaned_text):
        add("non_linguistic_text", cleaned_text[:80], "", "sin caracteres alfabeticos", "high")

    ratio = _garbage_ratio(cleaned_text)
    if ratio >= 0.25:
        add("high_garbage_ratio", cleaned_text[:80], "", f"garbage_ratio={ratio:.2f}", "high")
    elif ratio >= 0.12:
        add("medium_garbage_ratio", cleaned_text[:80], "", f"garbage_ratio={ratio:.2f}", "medium")

    for token in tokens:
        norm = normalizar_token(token.strip(".,;:!?¿¡()[]"))
        if len(norm) >= 30:
            add("extreme_long_token", token, "", f"token_len={len(norm)}", "high")
        elif len(norm) >= 20:
            add("long_suspicious_token", token, "", f"token_len={len(norm)}", "medium")
        if re.search(r"[^a-z0-9áéíóúüñÁÉÍÓÚÜÑ._'¿?¡!,-]", token):
            add("rare_character_token", token, "", "caracter fuera del set esperado", "medium")
        if re.search(r"([a-záéíóúüñ])\1{3,}", norm):
            add("repeated_character_token", token, "", "caracter repetido 4+ veces", "medium")
        if re.search(r"[bcdfghjklmnñpqrstvwxyz]{6,}", norm):
            add("consonant_run_token", token, "", "racha consonantica 6+", "low")

    for i in range(len(tokens) - 3):
        ventana = [normalizar_token(t.strip(".,;:!?¿¡()[]")) for t in tokens[i:i + 4]]
        if len(set(ventana)) == 1 and ventana[0]:
            add("anomalous_repetition", " ".join(tokens[i:i + 4]), "", "mismo token repetido 4 veces", "medium")
            break

    n_frames = int(float(row.get("n_frames") or 0))
    if n_frames > 0:
        dur = n_frames / 25.0
        wps = len(tokens) / max(dur, 0.01)
        if dur >= 1.0 and wps < 0.45:
            add("too_short_for_duration", cleaned_text[:80], "", f"words_per_second={wps:.2f}", "medium")
        elif wps > 6.5:
            add("too_long_for_duration", cleaned_text[:80], "", f"words_per_second={wps:.2f}", "medium")

    marker_tokens = {normalizar_token(t.strip("[]()")) for t in tokens}
    if marker_tokens and marker_tokens.issubset(NON_SPEECH_MARKERS):
        add("only_non_speech_markers", cleaned_text[:80], "", "solo markers no hablados", "high")

    return candidates


def clasificar_usabilidad(cleaned_text: str, candidates: list[dict[str, str]]) -> tuple[str, list[str]]:
    tipos = [c["candidate_type"] for c in candidates]
    if not cleaned_text:
        return "bad_candidate", ["empty_text"]
    severe = {"empty_text", "non_linguistic_text", "high_garbage_ratio", "extreme_long_token", "only_non_speech_markers"}
    if any(t in severe for t in tipos):
        return "bad_candidate", sorted(set(t for t in tipos if t in severe))
    if len(candidates) >= 3:
        return "questionable", ["many_review_candidates"] + sorted(set(tipos))
    if candidates:
        return "questionable", sorted(set(tipos))
    return "usable", ["auto_clean_safe_only_or_no_issue"]


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


def _write_split(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    escribir_csv(path, rows, fieldnames)


def _build_policy_index(policy_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["source_id"], row["clip"]): row for row in policy_rows}


def _write_stronger_splits(output_base: Path, split_rows: list[dict[str, str]], policy_rows: list[dict[str, str]]) -> dict[str, int]:
    idx = _build_policy_index(policy_rows)
    fieldnames = _fieldnames(split_rows)
    bad = {key for key, row in idx.items() if row["transcript_usability"] == "bad_candidate"}

    original_train = [r for r in split_rows if r["split"] == "train"]
    original_val = [r for r in split_rows if r["split"] == "val"]
    stronger_train = [r for r in original_train if (r["titulo"], r["clip"]) not in bad]
    stronger_val = list(original_val)

    visual_train_path = VISUAL_OUTPUT_BASE / "splits_visual_cleaned" / "train.csv"
    visual_val_path = VISUAL_OUTPUT_BASE / "splits_visual_cleaned" / "val.csv"
    visual_train = leer_csv(visual_train_path) if visual_train_path.exists() else []
    visual_val = leer_csv(visual_val_path) if visual_val_path.exists() else []
    all_train = [r for r in visual_train if (r.get("titulo", r.get("source_id", "")), r["clip"]) not in bad]
    all_val = list(visual_val)
    all_fieldnames = _fieldnames(visual_train) or fieldnames

    stronger_dir = output_base / "splits_transcript_cleaned_stronger"
    combined_dir = output_base / "splits_all_combined"
    _write_split(stronger_dir / "train.csv", stronger_train, fieldnames)
    _write_split(stronger_dir / "val.csv", stronger_val, fieldnames)
    _write_split(combined_dir / "train.csv", all_train, all_fieldnames)
    _write_split(combined_dir / "val.csv", all_val, all_fieldnames)

    return {
        "transcript_bad_candidates": len(bad),
        "splits_transcript_cleaned_stronger_train": len(stronger_train),
        "splits_transcript_cleaned_stronger_val": len(stronger_val),
        "splits_all_combined_train": len(all_train),
        "splits_all_combined_val": len(all_val),
    }


def build_transcript_overlays(
    splits_path: Path = DEFAULT_SPLITS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    repo_root: Path = REPO_ROOT,
    lexicon_path: Path = DEFAULT_LEXICON,
) -> dict[str, object]:
    rows = leer_csv(splits_path)
    lexicon = cargar_lexicon(lexicon_path)
    current_root = output_base / "transcripts_current"
    cleaned_root = output_base / "transcripts_cleaned_stronger"
    legacy_cleaned_root = output_base / "transcripts_cleaned_restricted"
    changes_csv = output_base / "transcript_cleaning_changes.csv"
    candidates_csv = output_base / "transcript_cleaning_candidates.csv"
    policy_csv = output_base / "transcript_quality_policy.csv"

    change_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    policy_rows: list[dict[str, str]] = []
    changed_count = 0

    for row in rows:
        source_id = row["titulo"]
        clip = row["clip"]
        original_text = row.get("texto", "")
        cleaned_text, changes, evidence_items = auto_clean_safe(original_text, source_id, lexicon)
        candidates = detectar_candidates(row, cleaned_text)
        usability, reasons = clasificar_usabilidad(cleaned_text, candidates)

        current_path = current_root / source_id / f"{clip}.txt"
        cleaned_path = cleaned_root / source_id / f"{clip}.txt"
        legacy_cleaned_path = legacy_cleaned_root / source_id / f"{clip}.txt"
        escribir_texto(current_path, original_text.strip())
        escribir_texto(cleaned_path, cleaned_text)
        escribir_texto(legacy_cleaned_path, cleaned_text)

        original_path = source_txt_path(row, repo_root)
        source_exists = original_path.exists()
        changed = cleaned_text != original_text
        changed_count += int(changed)
        change_rows.append(
            {
                "source_id": source_id,
                "clip": clip,
                "original_path": str(original_path),
                "cleaned_path": str(cleaned_path),
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "changed": "true" if changed else "false",
                "change_type": ";".join(changes) if changes else "none",
                "evidence": (
                    "; ".join(evidence_items)
                    if evidence_items
                    else f"source_txt_exists={source_exists}; sin cambios auto_clean_safe"
                ),
                "confidence": "high" if changed else "none",
                "auto_applied": "true" if changed else "false",
            }
        )
        candidate_rows.extend(candidates)
        policy_rows.append(
            {
                "source_id": source_id,
                "clip": clip,
                "split": row["split"],
                "transcript_usability": usability,
                "transcript_reasons": ";".join(reasons),
                "transcript_policy_moderate": "exclude" if usability == "bad_candidate" else "keep",
            }
        )

    escribir_csv(changes_csv, change_rows, CHANGE_COLUMNS)
    escribir_csv(candidates_csv, candidate_rows, CANDIDATE_COLUMNS)
    escribir_csv(policy_csv, policy_rows, QUALITY_COLUMNS)
    split_summary = _write_stronger_splits(output_base, rows, policy_rows)

    usability_counts = Counter(row["transcript_usability"] for row in policy_rows)
    candidate_counts = Counter(row["candidate_type"] for row in candidate_rows)
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
        "aggressive_candidates": len(candidate_rows),
        "candidate_counts": dict(candidate_counts),
        "transcript_usability_counts": dict(usability_counts),
        "excluded_by_policy_moderate": usability_counts.get("bad_candidate", 0),
        "current_root": str(current_root),
        "cleaned_root": str(cleaned_root),
        "changes_csv": str(changes_csv),
        "candidates_csv": str(candidates_csv),
        "policy_csv": str(policy_csv),
        "split_summary": split_summary,
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_transcript_overlays(args.splits, args.output_base, lexicon_path=args.lexicon)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
