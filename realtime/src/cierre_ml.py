"""Entrenamiento liviano para el cierre de oracion.

La idea de este modulo es dejar listo un clasificador barato para decidir
wait/commit/low_confidence sobre el buffer textual que produce el VSR. No usa
dependencias externas: sirve como baseline entrenable y como fallback de baja
latencia antes de probar modelos mas pesados en VM.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from realtime.src.cierre import HeuristicClosureProvider
from realtime.src.contracts import CommitAction, CommitDecision, PartialHypothesis
from realtime.src.validation import validate_commit_decision


LABELS = [CommitAction.WAIT.value, CommitAction.COMMIT.value, CommitAction.LOW_CONFIDENCE.value]
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(slots=True)
class ClosureTrainingExample:
    """Ejemplo causal: texto visible hasta un clip y accion esperada."""

    source_id: str
    clip_id: str
    partial_text: str
    expected_action: str
    committed_text: str = ""
    sentence_id: str = ""
    synthetic: bool = False
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_case(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_action"] = self.expected_action
        return data


class MajorityClosureProvider:
    """Baseline simple: siempre devuelve la clase mayoritaria del train."""

    def __init__(self, action: str = CommitAction.WAIT.value):
        self.action = CommitAction(action)
        self.name = f"majority_{self.action.value}"

    def decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        committed_text = hypothesis.partial_text if self.action == CommitAction.COMMIT else ""
        return CommitDecision(
            action=self.action,
            committed_text=committed_text,
            confidence=0.5,
            reason="majority_baseline",
            risk_flags=["baseline"],
        )


class FeatureExtractor:
    """Features discretas para un clasificador lineal muy rapido."""

    dangling_tokens = {
        "a",
        "al",
        "con",
        "de",
        "del",
        "el",
        "en",
        "la",
        "lo",
        "para",
        "pero",
        "porque",
        "por",
        "que",
        "si",
        "sin",
        "un",
        "una",
        "y",
    }
    dangling_suffixes = (
        "yo creo que",
        "creo que",
        "me parece que",
        "pienso que",
        "es que",
        "porque",
        "pero",
        "o sea",
        "entonces",
    )

    def __init__(self, *, include_heuristic: bool = False):
        self.include_heuristic = include_heuristic
        self._heuristic = HeuristicClosureProvider() if include_heuristic else None

    def extract(self, text: str, metadata: dict[str, Any] | None = None) -> dict[str, float]:
        metadata = metadata or {}
        normalized = " ".join((text or "").split()).strip().lower()
        tokens = _TOKEN_RE.findall(normalized)
        features: dict[str, float] = {"bias": 1.0}

        count = len(tokens)
        features[f"word_bin={_bucket(count, [1, 3, 5, 8, 12, 18])}"] = 1.0
        features[f"char_bin={_bucket(len(normalized), [8, 20, 40, 80, 140, 220])}"] = 1.0
        features[f"buffer_clip_bin={_bucket(int(metadata.get('buffer_clip_count') or 1), [1, 2, 3, 4, 6])}"] = 1.0

        if tokens:
            features[f"last={tokens[-1]}"] = 1.0
            features[f"first={tokens[0]}"] = 1.0
            if len(tokens) >= 2:
                features[f"last2={' '.join(tokens[-2:])}"] = 1.0
            if len(tokens) >= 3:
                features[f"last3={' '.join(tokens[-3:])}"] = 1.0
            for token in tokens[-16:]:
                features[f"tok={token}"] = features.get(f"tok={token}", 0.0) + 1.0

        if _ends_dangling(normalized, tokens, self.dangling_tokens, self.dangling_suffixes):
            features["ends_dangling"] = 1.0
        if _is_repetitive(tokens):
            features["repetitive"] = 1.0
        if any(token.isdigit() for token in tokens):
            features["has_number"] = 1.0
        if normalized.endswith((".", "?", "!")):
            features["has_punctuation_end"] = 1.0
        if metadata.get("synthetic"):
            features["is_synthetic"] = 1.0

        if self._heuristic is not None:
            decision = self._heuristic.decide(PartialHypothesis(partial_text=text))
            features[f"heuristic_action={decision.action.value}"] = 1.0
            features[f"heuristic_reason={decision.reason}"] = 1.0
            features[f"heuristic_conf={_bucket_float(decision.confidence, [0.25, 0.55, 0.75, 0.9])}"] = 1.0
            for flag in decision.risk_flags:
                features[f"heuristic_flag={flag}"] = 1.0

        return features


class LinearClosureProvider:
    """Provider entrenado con perceptron multiclase."""

    def __init__(
        self,
        *,
        weights: dict[str, dict[str, float]],
        include_heuristic: bool,
        labels: list[str] | None = None,
        name: str = "linear_closure",
        metadata: dict[str, Any] | None = None,
    ):
        self.weights = weights
        self.labels = labels or LABELS
        self.extractor = FeatureExtractor(include_heuristic=include_heuristic)
        self.include_heuristic = include_heuristic
        self.name = name
        self.metadata = metadata or {}

    def decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        features = self.extractor.extract(hypothesis.partial_text, hypothesis.metadata)
        scores = _scores(self.weights, self.labels, features)
        action_value = max(scores, key=scores.get)
        action = CommitAction(action_value)
        confidence = _softmax_confidence(scores, action_value)
        committed_text = hypothesis.partial_text if action == CommitAction.COMMIT else ""
        return CommitDecision(
            action=action,
            committed_text=committed_text,
            confidence=confidence,
            reason=f"{self.name}_score",
            risk_flags=["trained_linear"] if action == CommitAction.COMMIT else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": "linear",
            "name": self.name,
            "labels": self.labels,
            "include_heuristic": self.include_heuristic,
            "weights": self.weights,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "LinearClosureProvider":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights={label: {feat: float(value) for feat, value in weights.items()} for label, weights in data["weights"].items()},
            include_heuristic=bool(data.get("include_heuristic")),
            labels=[str(label) for label in data.get("labels", LABELS)],
            name=str(data.get("name") or "linear_closure"),
            metadata=dict(data.get("metadata") or {}),
        )


def load_training_examples(paths: Iterable[str | Path]) -> list[ClosureTrainingExample]:
    examples: list[ClosureTrainingExample] = []
    for path in _expand_input_paths(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "clip_decisions" in data:
            examples.extend(_examples_from_decisions(data))
        elif "clips" in data and "sentences" in data:
            examples.extend(_examples_from_sequence(data))
        else:
            raise ValueError(f"Formato de cierre no reconocido: {path}")
    if not examples:
        raise ValueError("No se cargaron ejemplos de cierre")
    return examples


def train_linear_model(
    examples: list[ClosureTrainingExample],
    *,
    include_heuristic: bool = False,
    class_balance: bool = False,
    epochs: int = 8,
    seed: int = 13,
    name: str = "linear_closure",
) -> LinearClosureProvider:
    extractor = FeatureExtractor(include_heuristic=include_heuristic)
    rng = random.Random(seed)
    weights = {label: {} for label in LABELS}
    class_weights = _class_weights(examples) if class_balance else {label: 1.0 for label in LABELS}

    training_rows = list(examples)
    for _ in range(epochs):
        rng.shuffle(training_rows)
        for example in training_rows:
            features = extractor.extract(example.partial_text, example.metadata)
            predicted = _predict_label(weights, LABELS, features)
            expected = example.expected_action
            if predicted == expected:
                continue
            amount = class_weights.get(expected, 1.0)
            _update(weights[expected], features, amount)
            _update(weights[predicted], features, -amount)

    metadata = {
        "epochs": epochs,
        "class_balance": class_balance,
        "train_examples": len(examples),
        "train_sources": sorted({example.source_id for example in examples}),
    }
    return LinearClosureProvider(
        weights=weights,
        include_heuristic=include_heuristic,
        labels=LABELS,
        name=name,
        metadata=metadata,
    )


def split_examples(
    examples: list[ClosureTrainingExample],
    *,
    seed: int = 13,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[ClosureTrainingExample]]:
    """Divide por fuente si hay suficientes fuentes; si no, por orden causal."""

    sources = sorted({example.source_id for example in examples})
    if len(sources) >= 3:
        rng = random.Random(seed)
        shuffled = list(sources)
        rng.shuffle(shuffled)
        train_n = max(1, round(len(shuffled) * train_ratio))
        val_n = max(1, round(len(shuffled) * val_ratio))
        train_sources = set(shuffled[:train_n])
        val_sources = set(shuffled[train_n : train_n + val_n])
        test_sources = set(shuffled[train_n + val_n :])
        if test_sources:
            return {
                "train": [example for example in examples if example.source_id in train_sources],
                "val": [example for example in examples if example.source_id in val_sources],
                "test": [example for example in examples if example.source_id in test_sources],
            }

    ordered = sorted(examples, key=lambda row: (row.source_id, row.order, row.clip_id))
    train_n = max(1, int(len(ordered) * train_ratio))
    val_n = max(1, int(len(ordered) * val_ratio))
    return {
        "train": ordered[:train_n],
        "val": ordered[train_n : train_n + val_n] or ordered[:train_n],
        "test": ordered[train_n + val_n :] or ordered[train_n : train_n + val_n] or ordered[:train_n],
    }


def evaluate_provider(provider, examples: list[ClosureTrainingExample]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    fallbacks = 0
    for example in examples:
        hypothesis = PartialHypothesis(
            partial_text=example.partial_text,
            segment_id=example.clip_id,
            source=example.source_id,
            metadata=example.metadata,
        )
        started = time.perf_counter()
        decision = provider.decide(hypothesis)
        latency_ms = (time.perf_counter() - started) * 1000.0
        decision, used_fallback = validate_commit_decision(decision)
        provider_fallback = "fallback" in decision.risk_flags
        fallbacks += int(used_fallback or provider_fallback)
        rows.append(
            {
                "source_id": example.source_id,
                "clip_id": example.clip_id,
                "expected": example.expected_action,
                "predicted": decision.action.value,
                "correct": example.expected_action == decision.action.value,
                "latency_ms": round(latency_ms, 4),
                "fallback": used_fallback or provider_fallback,
                "reason": decision.reason,
            }
        )
        latencies.append(latency_ms)

    metrics = _classification_metrics(rows)
    metrics.update(
        {
            "count": len(rows),
            "provider": getattr(provider, "name", provider.__class__.__name__),
            "latency_ms": _latency_summary(latencies),
            "fallbacks": fallbacks,
            "selection_score": _selection_score(metrics),
            "rows": rows,
        }
    )
    return metrics


def make_majority_provider(examples: list[ClosureTrainingExample]) -> MajorityClosureProvider:
    counts = {label: 0 for label in LABELS}
    for example in examples:
        counts[example.expected_action] = counts.get(example.expected_action, 0) + 1
    action = max(counts, key=counts.get)
    return MajorityClosureProvider(action)


def save_cases_jsonl(examples: list[ClosureTrainingExample], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_case(), ensure_ascii=False, sort_keys=True) + "\n")
    return output


def _examples_from_sequence(data: dict[str, Any]) -> list[ClosureTrainingExample]:
    source_id = str(data["source_id"])
    synthetic = bool(data.get("synthetic"))
    sentences = {str(row["commit_after_clip"]): row for row in data.get("sentences", [])}
    clips = data.get("clips", [])
    examples: list[ClosureTrainingExample] = []
    buffer_texts: list[str] = []
    buffer_count = 0
    for order, clip in enumerate(clips, start=1):
        clip_id = str(clip["clip_id"])
        text = str(clip.get("text") or clip.get("raw_text") or "")
        buffer_texts.append(text)
        buffer_count += 1
        partial_text = " ".join(part for part in buffer_texts if part).strip()
        sentence = sentences.get(clip_id)
        expected = CommitAction.COMMIT.value if sentence else CommitAction.WAIT.value
        examples.append(
            ClosureTrainingExample(
                source_id=source_id,
                clip_id=clip_id,
                partial_text=partial_text,
                expected_action=expected,
                committed_text=str(sentence.get("text") or partial_text) if sentence else "",
                sentence_id=str(sentence.get("sentence_id") or "") if sentence else "",
                synthetic=synthetic,
                order=order,
                metadata={"synthetic": synthetic, "buffer_clip_count": buffer_count},
            )
        )
        if sentence:
            buffer_texts = []
            buffer_count = 0
    return examples


def _examples_from_decisions(data: dict[str, Any]) -> list[ClosureTrainingExample]:
    source_id = str(data["source_id"])
    synthetic = bool(data.get("synthetic"))
    sentence_by_id = {str(row["sentence_id"]): row for row in data.get("sentences", [])}
    examples: list[ClosureTrainingExample] = []
    buffer_count = 0
    for order, decision in enumerate(data.get("clip_decisions", []), start=1):
        action = CommitAction(str(decision["action"])).value
        sentence_id = str(decision.get("committed_sentence_id") or "")
        sentence = sentence_by_id.get(sentence_id)
        buffer_count += 1
        examples.append(
            ClosureTrainingExample(
                source_id=source_id,
                clip_id=str(decision["clip_id"]),
                partial_text=str(decision.get("visible_context") or ""),
                expected_action=action,
                committed_text=str(sentence.get("text") or decision.get("visible_context") or "") if sentence else "",
                sentence_id=sentence_id,
                synthetic=synthetic,
                order=order,
                metadata={"synthetic": synthetic, "buffer_clip_count": buffer_count},
            )
        )
        if action == CommitAction.COMMIT.value:
            buffer_count = 0
    return examples


def _expand_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(path)
    return expanded


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    metrics: dict[str, Any] = {
        "accuracy": _safe_div(sum(1 for row in rows if row["correct"]), total),
        "per_action": {},
    }
    for label in LABELS:
        precision = _precision(rows, label)
        recall = _recall(rows, label)
        metrics["per_action"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": sum(1 for row in rows if row["expected"] == label),
        }
    metrics["macro_f1"] = round(sum(metrics["per_action"][label]["f1"] for label in LABELS) / len(LABELS), 4)
    metrics["commit_precision"] = metrics["per_action"][CommitAction.COMMIT.value]["precision"]
    metrics["commit_recall"] = metrics["per_action"][CommitAction.COMMIT.value]["recall"]
    metrics["commit_f1"] = metrics["per_action"][CommitAction.COMMIT.value]["f1"]
    metrics["premature_commit_rate"] = _safe_div(
        sum(1 for row in rows if row["predicted"] == CommitAction.COMMIT.value and row["expected"] != CommitAction.COMMIT.value),
        sum(1 for row in rows if row["expected"] != CommitAction.COMMIT.value),
    )
    metrics["unnecessary_wait_rate"] = _safe_div(
        sum(1 for row in rows if row["predicted"] == CommitAction.WAIT.value and row["expected"] == CommitAction.COMMIT.value),
        sum(1 for row in rows if row["expected"] == CommitAction.COMMIT.value),
    )
    metrics["low_confidence_recall"] = metrics["per_action"][CommitAction.LOW_CONFIDENCE.value]["recall"]
    return metrics


def _selection_score(metrics: dict[str, Any]) -> float:
    # Penaliza mas los commits tempranos porque cortan ideas y son caros de corregir.
    score = (
        metrics.get("commit_f1", 0.0)
        + 0.08 * metrics.get("macro_f1", 0.0)
        + 0.04 * metrics.get("accuracy", 0.0)
        - 0.45 * metrics.get("premature_commit_rate", 0.0)
        - 0.2 * metrics.get("unnecessary_wait_rate", 0.0)
    )
    return round(score, 4)


def _scores(weights: dict[str, dict[str, float]], labels: list[str], features: dict[str, float]) -> dict[str, float]:
    return {label: sum(weights.get(label, {}).get(name, 0.0) * value for name, value in features.items()) for label in labels}


def _predict_label(weights: dict[str, dict[str, float]], labels: list[str], features: dict[str, float]) -> str:
    scores = _scores(weights, labels, features)
    return max(scores, key=scores.get)


def _update(weights: dict[str, float], features: dict[str, float], amount: float) -> None:
    for name, value in features.items():
        new_value = weights.get(name, 0.0) + amount * value
        if abs(new_value) < 1e-12:
            weights.pop(name, None)
        else:
            weights[name] = new_value


def _class_weights(examples: list[ClosureTrainingExample]) -> dict[str, float]:
    counts = {label: 0 for label in LABELS}
    for example in examples:
        counts[example.expected_action] = counts.get(example.expected_action, 0) + 1
    total = sum(counts.values()) or 1
    return {label: total / (len(LABELS) * max(1, count)) for label, count in counts.items()}


def _precision(rows: list[dict[str, Any]], action: str) -> float:
    predicted = [row for row in rows if row["predicted"] == action]
    return _safe_div(sum(1 for row in predicted if row["expected"] == action), len(predicted))


def _recall(rows: list[dict[str, Any]], action: str) -> float:
    expected = [row for row in rows if row["expected"] == action]
    return _safe_div(sum(1 for row in expected if row["predicted"] == action), len(expected))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {"p50": round(statistics.median(ordered), 4), "p95": round(ordered[p95_index], 4)}


def _softmax_confidence(scores: dict[str, float], action: str) -> float:
    max_score = max(scores.values()) if scores else 0.0
    exps = {label: math.exp(max(-60.0, min(60.0, score - max_score))) for label, score in scores.items()}
    total = sum(exps.values()) or 1.0
    return round(exps.get(action, 0.0) / total, 4)


def _bucket(value: int, limits: list[int]) -> str:
    for limit in limits:
        if value <= limit:
            return f"<= {limit}"
    return f"> {limits[-1]}"


def _bucket_float(value: float, limits: list[float]) -> str:
    for limit in limits:
        if value <= limit:
            return f"<= {limit}"
    return f"> {limits[-1]}"


def _ends_dangling(text: str, tokens: list[str], dangling_tokens: set[str], dangling_suffixes: tuple[str, ...]) -> bool:
    if not tokens:
        return True
    if tokens[-1] in dangling_tokens:
        return True
    return any(text.endswith(suffix) for suffix in dangling_suffixes)


def _is_repetitive(tokens: list[str]) -> bool:
    if len(tokens) < 3:
        return False
    counts = {token: tokens.count(token) for token in set(tokens)}
    if max(counts.values()) / len(tokens) >= 0.6:
        return True
    run = 1
    for prev, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == prev else 1
        if run >= 3:
            return True
    return False
