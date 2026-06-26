"""Flujo causal por secuencias de clips para cierre de oracion."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from realtime.src.contracts import CommitAction, PartialHypothesis
from realtime.src.provider_factory import make_closure_provider
from realtime.src.validation import validate_commit_decision


@dataclass(slots=True)
class ClipText:
    """Texto asociado a un clip en orden temporal."""

    clip_id: str
    text: str
    order: int
    source_id: str = ""
    split: str = ""
    speaker: str = ""
    n_frames: int | None = None
    npz: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class GroundTruthSentence:
    """Oracion completa esperada sobre una secuencia de clips."""

    sentence_id: str
    text: str
    start_clip: str
    end_clip: str
    commit_after_clip: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class SequenceGroundTruth:
    """Ground truth oracional de una fuente o video."""

    source_id: str
    mode: str = "causal"
    clips: list[ClipText] = field(default_factory=list)
    sentences: list[GroundTruthSentence] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "mode": self.mode,
            "clips": [clip.to_dict() for clip in self.clips],
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "notes": self.notes,
        }


def load_sequence_ground_truth(path: str | Path) -> SequenceGroundTruth:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = [
        ClipText(
            clip_id=str(row["clip_id"]),
            text=str(row["text"]),
            order=int(row["order"]),
            source_id=str(row.get("source_id") or data.get("source_id") or ""),
            split=str(row.get("split") or ""),
            speaker=str(row.get("speaker") or ""),
            n_frames=_optional_int(row.get("n_frames")),
            npz=str(row.get("npz") or ""),
        )
        for row in data.get("clips", [])
    ]
    sentences = [
        GroundTruthSentence(
            sentence_id=str(row["sentence_id"]),
            text=str(row["text"]),
            start_clip=str(row["start_clip"]),
            end_clip=str(row["end_clip"]),
            commit_after_clip=str(row["commit_after_clip"]),
            notes=str(row.get("notes") or ""),
        )
        for row in data.get("sentences", [])
    ]
    sequence = SequenceGroundTruth(
        source_id=str(data["source_id"]),
        mode=str(data.get("mode") or "causal"),
        clips=clips,
        sentences=sentences,
        notes=str(data.get("notes") or ""),
    )
    validate_sequence_ground_truth(sequence)
    return sequence


def validate_sequence_ground_truth(sequence: SequenceGroundTruth) -> None:
    if not sequence.source_id:
        raise ValueError("source_id es obligatorio")
    if not sequence.clips:
        raise ValueError("clips no puede estar vacio")
    clip_ids = [clip.clip_id for clip in sequence.clips]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("clip_id duplicado en clips")
    known = set(clip_ids)
    for sentence in sequence.sentences:
        if sentence.start_clip not in known:
            raise ValueError(f"start_clip desconocido: {sentence.start_clip}")
        if sentence.end_clip not in known:
            raise ValueError(f"end_clip desconocido: {sentence.end_clip}")
        if sentence.commit_after_clip not in known:
            raise ValueError(f"commit_after_clip desconocido: {sentence.commit_after_clip}")


def load_clips_from_split(
    split_path: str | Path,
    *,
    source_id: str | None = None,
    limit: int | None = None,
) -> list[ClipText]:
    """Carga textos ordenados desde un split CSV sin abrir ROIs ni videos."""

    path = Path(split_path)
    rows: list[ClipText] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        filtered = [row for row in reader if source_id is None or row.get("titulo") == source_id]
    filtered.sort(key=lambda row: (_clip_number(row.get("clip", "")), row.get("clip", "")))
    for order, row in enumerate(filtered[:limit], start=1):
        rows.append(
            ClipText(
                clip_id=str(row.get("clip") or f"clip_{order:04d}"),
                text=str(row.get("texto") or ""),
                order=order,
                source_id=str(row.get("titulo") or source_id or path.stem),
                split=str(row.get("split") or path.stem),
                speaker=str(row.get("spk") or ""),
                n_frames=_optional_int(row.get("n_frames")),
                npz=str(row.get("npz") or ""),
            )
        )
    return rows


def export_llm_annotation_packet(
    clips: Iterable[ClipText],
    *,
    source_id: str,
    output_path: str | Path,
) -> Path:
    """Exporta un paquete Markdown para anotar oraciones con un LLM potente."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clip_list = list(clips)
    lines = [
        f"# Ground truth oracional para {source_id}",
        "",
        "Objetivo: separar el texto continuo en oraciones completas, preservando el orden de clips.",
        "",
        "Reglas:",
        "- No agregar informacion nueva.",
        "- No corregir agresivamente: solo puntuacion y segmentacion oracional.",
        "- Si una oracion usa varios clips, marcar start_clip y end_clip.",
        "- commit_after_clip debe ser el ultimo clip necesario para saber que la oracion termino.",
        "- Devolver JSON valido con el esquema indicado.",
        "",
        "Esquema esperado:",
        "```json",
        json.dumps(
            {
                "source_id": source_id,
                "mode": "causal",
                "sentences": [
                    {
                        "sentence_id": "s001",
                        "text": "Oracion completa.",
                        "start_clip": "clip_0000",
                        "end_clip": "clip_0002",
                        "commit_after_clip": "clip_0002",
                        "notes": "",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "Clips ordenados:",
        "",
    ]
    for clip in clip_list:
        lines.append(f"- {clip.clip_id}: {clip.text}")
    lines.extend(
        [
            "",
            "Texto continuo de referencia:",
            "",
            " ".join(clip.text for clip in clip_list if clip.text).strip(),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def merge_annotation_with_clips(
    clips: list[ClipText],
    annotation_path: str | Path,
    *,
    output_path: str | Path,
    source_id: str,
) -> Path:
    """Combina la respuesta del LLM con clips para producir ground truth completo."""

    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    payload = {
        "source_id": str(data.get("source_id") or source_id),
        "mode": str(data.get("mode") or "causal"),
        "clips": [clip.to_dict() for clip in clips],
        "sentences": data.get("sentences", []),
        "notes": str(data.get("notes") or "Anotacion asistida; revisar manualmente antes de usar como benchmark final."),
    }
    sequence = load_sequence_ground_truth_from_dict(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sequence.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def load_sequence_ground_truth_from_dict(data: dict[str, object]) -> SequenceGroundTruth:
    # Evita duplicar el parseo publico sin depender de archivos temporales externos.
    clips = [
        ClipText(
            clip_id=str(row["clip_id"]),
            text=str(row["text"]),
            order=int(row["order"]),
            source_id=str(row.get("source_id") or data.get("source_id") or ""),
            split=str(row.get("split") or ""),
            speaker=str(row.get("speaker") or ""),
            n_frames=_optional_int(row.get("n_frames")),
            npz=str(row.get("npz") or ""),
        )
        for row in data.get("clips", [])  # type: ignore[union-attr]
    ]
    sentences = [
        GroundTruthSentence(
            sentence_id=str(row["sentence_id"]),
            text=str(row["text"]),
            start_clip=str(row["start_clip"]),
            end_clip=str(row["end_clip"]),
            commit_after_clip=str(row["commit_after_clip"]),
            notes=str(row.get("notes") or ""),
        )
        for row in data.get("sentences", [])  # type: ignore[union-attr]
    ]
    sequence = SequenceGroundTruth(
        source_id=str(data["source_id"]),
        mode=str(data.get("mode") or "causal"),
        clips=clips,
        sentences=sentences,
        notes=str(data.get("notes") or ""),
    )
    validate_sequence_ground_truth(sequence)
    return sequence


def evaluate_causal_sequence(sequence: SequenceGroundTruth, provider) -> dict[str, object]:
    """Evalua cierre causal acumulando clips hasta cada commit."""

    expected_commit_clip = {sentence.commit_after_clip: sentence for sentence in sequence.sentences}
    clip_order = {clip.clip_id: clip.order for clip in sequence.clips}
    pending_sentences = list(sequence.sentences)
    buffer_texts: list[str] = []
    buffer_start_clip = ""
    rows = []
    early_commits = 0
    late_waits = 0
    correct_commits = 0
    unexpected_low_confidence = 0
    fallbacks = 0
    latencies: list[float] = []

    for clip in sequence.clips:
        if not buffer_texts:
            buffer_start_clip = clip.clip_id
        buffer_texts.append(clip.text)
        buffer = " ".join(part for part in buffer_texts if part).strip()
        row_buffer_start_clip = buffer_start_clip
        hypothesis = PartialHypothesis(
            partial_text=buffer,
            segment_id=clip.clip_id,
            source=sequence.source_id,
            metadata={"buffer_start_clip": buffer_start_clip, "current_clip": clip.clip_id},
        )
        started = time.perf_counter()
        decision = provider.decide(hypothesis)
        latency_ms = (time.perf_counter() - started) * 1000.0
        decision, fallback_used = validate_commit_decision(decision)
        provider_fallback = "fallback" in decision.risk_flags
        fallbacks += int(fallback_used or provider_fallback)
        latencies.append(latency_ms)

        expected_sentence = expected_commit_clip.get(clip.clip_id)
        expected_action = CommitAction.COMMIT if expected_sentence else CommitAction.WAIT
        if decision.action == CommitAction.COMMIT and expected_action == CommitAction.COMMIT:
            correct_commits += 1
            if pending_sentences and pending_sentences[0].commit_after_clip == clip.clip_id:
                pending_sentences.pop(0)
            buffer_texts = []
            buffer_start_clip = ""
        elif decision.action == CommitAction.COMMIT and expected_action != CommitAction.COMMIT:
            early_commits += 1
            buffer_texts = []
            buffer_start_clip = ""
        elif decision.action != CommitAction.COMMIT and expected_action == CommitAction.COMMIT:
            late_waits += 1
        if decision.action == CommitAction.LOW_CONFIDENCE and expected_action == CommitAction.WAIT:
            unexpected_low_confidence += 1

        rows.append(
            {
                "clip_id": clip.clip_id,
                "buffer_start_clip": row_buffer_start_clip,
                "buffer_text": buffer,
                "expected_action": expected_action.value,
                "expected_sentence_id": expected_sentence.sentence_id if expected_sentence else "",
                "predicted_action": decision.action.value,
                "committed_text": decision.committed_text,
                "reason": decision.reason,
                "risk_flags": decision.risk_flags,
                "latency_ms": round(latency_ms, 4),
                "fallback": fallback_used or provider_fallback,
            }
        )

    expected_commits = len(sequence.sentences)
    predicted_commits = sum(1 for row in rows if row["predicted_action"] == "commit")
    return {
        "source_id": sequence.source_id,
        "mode": sequence.mode,
        "clips": len(sequence.clips),
        "expected_commits": expected_commits,
        "predicted_commits": predicted_commits,
        "correct_commits": correct_commits,
        "early_commits": early_commits,
        "late_waits": late_waits,
        "missing_commits": max(0, expected_commits - correct_commits),
        "unexpected_low_confidence": unexpected_low_confidence,
        "fallbacks": fallbacks,
        "commit_precision": _safe_div(correct_commits, predicted_commits),
        "commit_recall": _safe_div(correct_commits, expected_commits),
        "latency_ms": _latency_summary(latencies),
        "rows": rows,
        "clip_order": clip_order,
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(ordered[len(ordered) // 2], 4),
        "p95": round(ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))], 4),
    }


def _safe_div(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


def _clip_number(clip_id: str) -> int:
    digits = "".join(ch for ch in clip_id if ch.isdigit())
    return int(digits) if digits else 0


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Herramientas secuenciales de cierre causal.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-annotation", help="Exporta clips ordenados para anotar con LLM.")
    export_parser.add_argument("--split", required=True)
    export_parser.add_argument("--source-id", required=True)
    export_parser.add_argument("--limit", type=int, default=40)
    export_parser.add_argument("--output", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evalua una secuencia con ground truth.")
    eval_parser.add_argument("--ground-truth", required=True)
    eval_parser.add_argument("--provider", choices=["heuristic", "ollama"], default="heuristic")
    eval_parser.add_argument("--ollama-model", default="qwen3:4b")
    eval_parser.add_argument("--ollama-url", default="http://localhost:11434")
    eval_parser.add_argument("--timeout-s", type=float, default=2.5)

    args = parser.parse_args()
    if args.command == "export-annotation":
        clips = load_clips_from_split(args.split, source_id=args.source_id, limit=args.limit)
        output = export_llm_annotation_packet(clips, source_id=args.source_id, output_path=args.output)
        print(json.dumps({"output": str(output), "clips": len(clips)}, ensure_ascii=False, sort_keys=True))
        return

    sequence = load_sequence_ground_truth(args.ground_truth)
    provider = make_closure_provider(
        args.provider,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(evaluate_causal_sequence(sequence, provider), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
