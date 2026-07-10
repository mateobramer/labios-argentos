"""Transcript cleaning conservador con evidencia ASR2 opcional."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr" / "splits" / "splits.csv"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr"
DEFAULT_LEXICON = REPO_ROOT / "cleaning/visual_quality" / "resources" / "entity_lexicon.csv"
DEFAULT_ASR2 = DEFAULT_OUTPUT_BASE / "transcript_second_pass_asr.csv"
DEFAULT_ASR_DISAGREEMENT = DEFAULT_OUTPUT_BASE / "transcript_asr_disagreement.csv"
VISUAL_OUTPUT_BASE = REPO_ROOT / "evaluation" / "outputs" / "visual_cleaning"

NON_SPEECH_MARKERS = {
    "musica",
    "aplausos",
    "risas",
    "subtitulos",
    "suscribete",
}

CHANGE_COLUMNS = [
    "source_id",
    "clip",
    "original_path",
    "cleaned_path",
    "original_text",
    "asr2_text",
    "cleaned_text",
    "changed",
    "change_type",
    "evidence",
    "confidence",
    "auto_applied",
    "asr_disagreement_level",
]

CANDIDATE_COLUMNS = [
    "source_id",
    "clip",
    "current_text",
    "asr2_text",
    "span",
    "suggestion",
    "candidate_type",
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
    token = unicodedata.normalize("NFKD", str(token).lower())
    return "".join(ch for ch in token if not unicodedata.combining(ch))


def normalizar_texto(texto: str) -> str:
    return " ".join(str(texto or "").strip().split())


def cargar_lexicon(path: Path = DEFAULT_LEXICON) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = leer_csv(path)
    for row in rows:
        row.setdefault("type", "other")
        row.setdefault("notes", "")
    return rows


def cargar_indice(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = leer_csv(path)
    return {(row.get("source_id", ""), row.get("clip", "")): row for row in rows}


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;|]", str(value or "")) if part.strip()]


def _entry_aliases(entry: dict[str, str]) -> list[str]:
    aliases = _split_list(entry.get("aliases", ""))
    canonical = entry.get("canonical", "").strip()
    return [a for a in aliases if a] + ([canonical] if canonical else [])


def _entry_hints(entry: dict[str, str]) -> list[str]:
    return [normalizar_texto(h) for h in _split_list(entry.get("source_hint", ""))]


def _distancia_chica(a: str, b: str) -> bool:
    a_norm = normalizar_token(a)
    b_norm = normalizar_token(b)
    if not a_norm or not b_norm:
        return False
    if abs(len(a_norm) - len(b_norm)) > max(2, int(len(b_norm) * 0.5)):
        return False
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.72


def _tokens(texto: str) -> list[str]:
    return [t for t in re.split(r"\s+", str(texto).strip()) if t]


def _norm_tokens(texto: str) -> list[str]:
    return [normalizar_token(t.strip(".,;:!?¿¡()[]")) for t in _tokens(texto)]


def _contains_phrase(texto: str, phrase: str) -> bool:
    phrase_tokens = _norm_tokens(phrase)
    if not phrase_tokens:
        return False
    tokens = _norm_tokens(texto)
    n = len(phrase_tokens)
    return any(tokens[i : i + n] == phrase_tokens for i in range(0, len(tokens) - n + 1))


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


def _source_or_asr_evidence(entry: dict[str, str], source_id: str, asr2_text: str) -> tuple[bool, list[str]]:
    canonical = entry.get("canonical", "").strip()
    source_norm = normalizar_texto(source_id).lower()
    asr_norm = normalizar_texto(asr2_text).lower()
    evidence: list[str] = []
    if canonical and _contains_phrase(asr_norm, canonical):
        evidence.append("canonical_in_asr2")
    hints = _entry_hints(entry)
    if hints and any(h.lower() in source_norm for h in hints):
        evidence.append("source_hint")
    if canonical and _contains_phrase(source_norm, canonical):
        evidence.append("canonical_in_source")
    return bool(evidence), evidence


def _replace_local_phrase(
    texto: str,
    entry: dict[str, str],
    source_id: str,
    asr2_text: str,
) -> tuple[str, list[str], list[str]]:
    canonical = entry.get("canonical", "").strip()
    if not canonical:
        return texto, [], []
    has_evidence, evidence = _source_or_asr_evidence(entry, source_id, asr2_text)
    if not has_evidence:
        return texto, [], []
    tokens = texto.split()
    norm_tokens = [normalizar_token(t.strip(".,;:!?¿¡()[]")) for t in tokens]
    cambios: list[str] = []
    for alias in _entry_aliases(entry):
        if normalizar_token(alias) == normalizar_token(canonical):
            continue
        alias_tokens = _norm_tokens(alias)
        canonical_tokens = _tokens(canonical)
        if not (1 <= len(alias_tokens) <= 4 and 1 <= len(canonical_tokens) <= 4):
            continue
        if abs(len(alias_tokens) - len(canonical_tokens)) > 2 and not _distancia_chica(alias, canonical):
            continue
        n = len(alias_tokens)
        i = 0
        while i <= len(tokens) - n:
            if norm_tokens[i : i + n] == alias_tokens:
                tokens = tokens[:i] + canonical_tokens + tokens[i + n :]
                norm_tokens = [normalizar_token(t.strip(".,;:!?¿¡()[]")) for t in tokens]
                cambios.append(f"{alias}->{canonical}")
                i += len(canonical_tokens)
            else:
                i += 1
    return (" ".join(tokens) if cambios else texto), cambios, evidence


def _aplicar_entidades_con_evidencia(
    texto: str,
    source_id: str,
    asr2_text: str,
    lexicon: list[dict[str, str]],
) -> tuple[str, list[str], list[str]]:
    cambios: list[str] = []
    evidence: list[str] = []
    salida = texto
    for entry in lexicon:
        replaced, entry_changes, entry_evidence = _replace_local_phrase(salida, entry, source_id, asr2_text)
        if entry_changes:
            salida = replaced
            cambios.extend(entry_changes)
            evidence.extend(entry_evidence)
    return salida, cambios, sorted(set(evidence))


def auto_clean_safe(
    texto: str,
    source_id: str = "",
    lexicon: list[dict[str, str]] | None = None,
    asr2_text: str = "",
) -> tuple[str, list[str], list[str]]:
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

    con_entidades, entity_changes, entity_evidence = _aplicar_entidades_con_evidencia(
        sin_markers,
        source_id,
        asr2_text,
        lexicon,
    )
    if entity_changes:
        cambios.append("entity_replacement_high_confidence")
        evidence.append("lexicon+asr2/source: " + "|".join(entity_changes[:5]))
        evidence.extend(entity_evidence)

    espacios = re.sub(r"\s+", " ", con_entidades).strip()
    if espacios != con_entidades:
        cambios.append("space_normalization")
        evidence.append("espacios multiples/strip")

    return espacios, cambios, evidence


def limpiar_restringido(texto: str) -> tuple[str, list[str]]:
    cleaned, changes, _ = auto_clean_safe(texto)
    return cleaned, changes


def _garbage_ratio(texto: str) -> float:
    if not texto:
        return 0.0
    valid = 0
    allowed = "áéíóúüñÁÉÍÓÚÜÑ-_'"
    for ch in texto:
        if ch.isalnum() or ch.isspace() or ch in allowed:
            valid += 1
    return 1.0 - (valid / len(texto))


def _candidate(
    source_id: str,
    clip: str,
    current_text: str,
    asr2_text: str,
    span: str,
    suggestion: str,
    candidate_type: str,
    evidence: str,
    confidence: str,
    auto_applied: str = "false",
    reason_not_auto_applied: str = "requiere revision humana; no hay evidencia fuerte para reescribir",
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "clip": clip,
        "current_text": current_text,
        "asr2_text": asr2_text,
        "span": span,
        "suggestion": suggestion,
        "candidate_type": candidate_type,
        "evidence": evidence,
        "confidence": confidence,
        "auto_applied": auto_applied,
        "reason_not_auto_applied": "" if auto_applied == "true" else reason_not_auto_applied,
    }


def detectar_candidates_basicos(row: dict[str, str], cleaned_text: str, asr2_text: str = "") -> list[dict[str, str]]:
    source_id = row["titulo"]
    clip = row["clip"]
    tokens = _tokens(cleaned_text)
    candidates: list[dict[str, str]] = []

    def add(candidate_type: str, span: str, suggestion: str, evidence: str, confidence: str = "medium") -> None:
        candidates.append(_candidate(source_id, clip, row.get("texto", ""), asr2_text, span, suggestion, candidate_type, evidence, confidence))

    if not cleaned_text:
        add("possible_empty_or_no_speech", "", "", "texto vacio despues de limpieza", "high")
        return candidates

    if not any(ch.isalpha() for ch in cleaned_text):
        add("possible_audio_text_mismatch", cleaned_text[:80], "", "sin caracteres alfabeticos", "high")

    ratio = _garbage_ratio(cleaned_text)
    if ratio >= 0.25:
        add("possible_audio_text_mismatch", cleaned_text[:80], "", f"garbage_ratio={ratio:.2f}", "high")
    elif ratio >= 0.12:
        add("asr_disagreement", cleaned_text[:80], "", f"garbage_ratio={ratio:.2f}", "medium")

    for token in tokens:
        norm = normalizar_token(token.strip(".,;:!?¿¡()[]"))
        if len(norm) >= 30:
            add("possible_audio_text_mismatch", token, "", f"token_len={len(norm)}", "high")
        elif len(norm) >= 20:
            add("asr_disagreement", token, "", f"token_len={len(norm)}", "medium")
        if re.search(r"[^a-z0-9áéíóúüñÁÉÍÓÚÜÑ._'¿?¡!,-]", token):
            add("asr_disagreement", token, "", "caracter fuera del set esperado", "medium")
        if re.search(r"([a-záéíóúüñ])\1{3,}", norm):
            add("asr_disagreement", token, "", "caracter repetido 4+ veces", "medium")

    for i in range(len(tokens) - 3):
        ventana = [normalizar_token(t.strip(".,;:!?¿¡()[]")) for t in tokens[i : i + 4]]
        if len(set(ventana)) == 1 and ventana[0]:
            add("possible_hallucination", " ".join(tokens[i : i + 4]), "", "mismo token repetido 4 veces", "medium")
            break

    n_frames = int(float(row.get("n_frames") or 0))
    if n_frames > 0:
        dur = n_frames / 25.0
        wps = len(tokens) / max(dur, 0.01)
        if dur >= 1.0 and wps < 0.45:
            add("possible_audio_text_mismatch", cleaned_text[:80], "", f"words_per_second={wps:.2f}", "medium")
        elif wps > 6.5:
            add("possible_audio_text_mismatch", cleaned_text[:80], "", f"words_per_second={wps:.2f}", "medium")

    marker_tokens = {normalizar_token(t.strip("[]()")) for t in tokens}
    if marker_tokens and marker_tokens.issubset(NON_SPEECH_MARKERS):
        add("possible_empty_or_no_speech", cleaned_text[:80], "", "solo markers no hablados", "high")

    return candidates


def detectar_entity_candidates(
    row: dict[str, str],
    cleaned_text: str,
    asr2_text: str,
    lexicon: list[dict[str, str]],
    auto_changes: list[str],
) -> list[dict[str, str]]:
    source_id = row["titulo"]
    clip = row["clip"]
    candidates: list[dict[str, str]] = []
    auto_joined = "|".join(auto_changes)
    current_text = row.get("texto", "")
    for entry in lexicon:
        canonical = entry.get("canonical", "").strip()
        if not canonical:
            continue
        entry_type = entry.get("type", "other")
        candidate_type = "slang_replacement_candidate" if entry_type == "slang" else "entity_replacement_candidate"
        has_evidence, evidence = _source_or_asr_evidence(entry, source_id, asr2_text)
        for alias in _entry_aliases(entry):
            if normalizar_token(alias) == normalizar_token(canonical):
                continue
            if _contains_phrase(current_text, alias) or _contains_phrase(cleaned_text, alias):
                auto = f"{alias}->{canonical}" in auto_joined
                candidates.append(
                    _candidate(
                        source_id,
                        clip,
                        row.get("texto", ""),
                        asr2_text,
                        alias,
                        canonical,
                        candidate_type,
                        ";".join(evidence) if evidence else "lexicon_alias_without_asr2_or_source_confirmation",
                        "high" if auto and has_evidence else "medium",
                        "true" if auto else "false",
                        "" if auto else "falta evidencia ASR2/source fuerte para autoaplicar",
                    )
                )
    return candidates


def candidates_por_disagreement(
    row: dict[str, str],
    asr_row: dict[str, str],
    disagreement_row: dict[str, str],
) -> list[dict[str, str]]:
    if not disagreement_row:
        return []
    level = disagreement_row.get("disagreement_level", "")
    if level in {"", "low"}:
        return []
    reasons = disagreement_row.get("reasons", "")
    if "asr2_blocked" in reasons or level == "blocked":
        candidate_type = "asr_disagreement"
        confidence = "low"
    elif "possible_timestamp_misalignment" in reasons:
        candidate_type = "possible_misalignment"
        confidence = "high" if level == "high" else "medium"
    elif "possible_whisper_hallucination" in reasons:
        candidate_type = "possible_hallucination"
        confidence = "high" if level == "high" else "medium"
    elif "possible_audio_text_mismatch" in reasons:
        candidate_type = "possible_audio_text_mismatch"
        confidence = "high" if level == "high" else "medium"
    else:
        candidate_type = "asr_disagreement"
        confidence = "high" if level == "high" else "medium"
    return [
        _candidate(
            row["titulo"],
            row["clip"],
            row.get("texto", ""),
            asr_row.get("asr2_text", ""),
            disagreement_row.get("token_diff_summary", "")[:120],
            "",
            candidate_type,
            f"level={level}; reasons={reasons}; wer={disagreement_row.get('wer_current_vs_asr2', '')}; cer={disagreement_row.get('cer_current_vs_asr2', '')}",
            confidence,
        )
    ]


def clasificar_usabilidad(
    cleaned_text: str,
    candidates: list[dict[str, str]],
    disagreement_row: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    tipos = [c["candidate_type"] for c in candidates if c.get("auto_applied") != "true"]
    if not cleaned_text:
        return "bad_candidate", ["empty_text"]
    level = (disagreement_row or {}).get("disagreement_level", "")
    reasons_raw = (disagreement_row or {}).get("reasons", "")
    reasons = [r for r in reasons_raw.split(";") if r]
    if level == "high":
        return "bad_candidate", sorted(set(reasons or ["high_asr2_disagreement"]))
    severe = {"possible_empty_or_no_speech", "possible_audio_text_mismatch"}
    severe_high = {c["candidate_type"] for c in candidates if c.get("confidence") == "high" and c["candidate_type"] in severe}
    if severe_high:
        return "bad_candidate", sorted(severe_high)
    if level == "medium":
        return "questionable", sorted(set(reasons or ["medium_asr2_disagreement"]))
    unresolved_entity = {
        c["candidate_type"]
        for c in candidates
        if c["candidate_type"] in {"entity_replacement_candidate", "slang_replacement_candidate"} and c.get("auto_applied") != "true"
    }
    if unresolved_entity:
        return "questionable", sorted(unresolved_entity)
    if len(tipos) >= 3:
        return "questionable", ["many_review_candidates"] + sorted(set(tipos))
    if tipos:
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


def _asr2_status(asr2_idx: dict[tuple[str, str], dict[str, str]]) -> str:
    if not asr2_idx:
        return "missing"
    statuses = Counter(row.get("status", "") for row in asr2_idx.values())
    if statuses.get("ok", 0):
        return "available"
    if statuses.get("blocked", 0) == len(asr2_idx):
        return "blocked"
    return "partial"


def build_transcript_overlays(
    splits_path: Path = DEFAULT_SPLITS,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    repo_root: Path = REPO_ROOT,
    lexicon_path: Path = DEFAULT_LEXICON,
    asr2_path: Path | None = None,
    asr_disagreement_path: Path | None = None,
) -> dict[str, object]:
    rows = leer_csv(splits_path)
    lexicon = cargar_lexicon(lexicon_path)
    asr2_idx = cargar_indice(asr2_path if asr2_path and asr2_path.exists() else None)
    disagreement_idx = cargar_indice(
        asr_disagreement_path if asr_disagreement_path and asr_disagreement_path.exists() else None
    )
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
    auto_replacements = 0

    for row in rows:
        source_id = row["titulo"]
        clip = row["clip"]
        key = (source_id, clip)
        original_text = row.get("texto", "")
        asr_row = asr2_idx.get(key, {})
        disagreement_row = disagreement_idx.get(key, {})
        asr2_text = asr_row.get("asr2_text", "")
        cleaned_text, changes, evidence_items = auto_clean_safe(original_text, source_id, lexicon, asr2_text)
        auto_replacements += int("entity_replacement_high_confidence" in changes)
        candidates = []
        candidates.extend(detectar_candidates_basicos(row, cleaned_text, asr2_text))
        candidates.extend(detectar_entity_candidates(row, cleaned_text, asr2_text, lexicon, evidence_items))
        candidates.extend(candidates_por_disagreement(row, asr_row, disagreement_row))
        usability, reasons = clasificar_usabilidad(cleaned_text, candidates, disagreement_row)

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
                "asr2_text": asr2_text,
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
                "asr_disagreement_level": disagreement_row.get("disagreement_level", ""),
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
    disagreement_counts = Counter(row.get("asr_disagreement_level", "") for row in change_rows)
    examples = [
        {
            "source_id": row["source_id"],
            "clip": row["clip"],
            "change_type": row["change_type"],
            "original_text": row["original_text"],
            "asr2_text": row["asr2_text"],
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
        "asr2_status": _asr2_status(asr2_idx),
        "asr2_rows": len(asr2_idx),
        "disagreement_rows": len(disagreement_idx),
        "disagreement_counts": dict(disagreement_counts),
        "replacement_candidates": len(candidate_rows),
        "auto_replacements": auto_replacements,
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
    ap.add_argument("--asr2", type=Path, default=None)
    ap.add_argument("--asr-disagreement", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_transcript_overlays(
        args.splits,
        args.output_base,
        lexicon_path=args.lexicon,
        asr2_path=args.asr2,
        asr_disagreement_path=args.asr_disagreement,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
