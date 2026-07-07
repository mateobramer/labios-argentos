"""Valida JSONL de patches LLM antes de aplicar clean_v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path


VALID_STATUS = {"unchanged", "patched", "needs_review", "bad_candidate"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_PATCH_TYPES = {
    "entity_fix",
    "slang_fix",
    "asr_word_fix",
    "punctuation_minimal",
    "other",
}


def word_count(text: str) -> int:
    return len(str(text or "").split())


def validate_record(record: dict, line_no: int) -> list[str]:
    errors: list[str] = []
    for field in ["clip_id", "large_text", "turbo_text", "clean_text", "status", "confidence", "patches"]:
        if field not in record:
            errors.append(f"line {line_no}: missing field {field}")

    status = record.get("status")
    if status not in VALID_STATUS:
        errors.append(f"line {line_no}: invalid status {status!r}")
    if record.get("confidence") not in VALID_CONFIDENCE:
        errors.append(f"line {line_no}: invalid confidence {record.get('confidence')!r}")

    clean_text = str(record.get("clean_text", "")).strip()
    large_text = str(record.get("large_text", "")).strip()
    if status in {"unchanged", "patched"} and not clean_text:
        errors.append(f"line {line_no}: clean_text empty for {status}")

    patches = record.get("patches")
    if not isinstance(patches, list):
        errors.append(f"line {line_no}: patches must be a list")
        patches = []
    if status == "patched" and not patches:
        errors.append(f"line {line_no}: patched requires at least one patch")
    if status == "unchanged" and clean_text and large_text and clean_text != large_text:
        errors.append(f"line {line_no}: unchanged must preserve large_text exactly")

    large_words = word_count(large_text)
    clean_words = word_count(clean_text)
    if status == "patched" and large_words >= 6 and clean_words < max(1, int(large_words * 0.55)):
        errors.append(f"line {line_no}: clean_text deletes too many words")

    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"line {line_no}: patch {index} must be object")
            continue
        if patch.get("patch_type") not in VALID_PATCH_TYPES:
            errors.append(f"line {line_no}: patch {index} invalid patch_type")
        evidence = patch.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"line {line_no}: patch {index} requires evidence")
        if not patch.get("span_before") or not patch.get("span_after"):
            errors.append(f"line {line_no}: patch {index} requires span_before/span_after")

    return errors


def validate_jsonl(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_no}: JSONL item must be object")
                continue
            errors.extend(validate_record(record, line_no))
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Uso: python -m data_cleaning_clean_v1.src.validate_patches patches.jsonl")
        return 2
    path = Path(argv[1])
    errors = validate_jsonl(path)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
