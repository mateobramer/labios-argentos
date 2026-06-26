"""Cierre de oracion conservador para el MVP realtime."""

from __future__ import annotations

import re
from collections import Counter

from realtime.src.contracts import CommitAction, CommitDecision, PartialHypothesis
from realtime.src.validation import fallback_commit, validate_commit_decision


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HeuristicClosureProvider:
    """Provider sin dependencias externas.

    La regla central es conservadora: si hay duda, espera. Esto evita commits
    prematuros y deja al futuro LLM competir contra una baseline segura.
    """

    name = "heuristic_closure"

    dangling_tokens = {
        "a",
        "al",
        "con",
        "de",
        "del",
        "el",
        "en",
        "la",
        "las",
        "lo",
        "los",
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
        "diria que",
        "es que",
        "porque",
        "pero",
    )
    sentence_endings = (".", "?", "!")

    def decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        try:
            decision = self._decide(hypothesis)
        except Exception:
            decision = fallback_commit("heuristic_exception")
        validated, _ = validate_commit_decision(decision)
        return validated

    def _decide(self, hypothesis: PartialHypothesis) -> CommitDecision:
        text = " ".join((hypothesis.partial_text or "").split()).strip()
        tokens = self._tokens(text)

        if not text or not tokens:
            return CommitDecision(
                action=CommitAction.LOW_CONFIDENCE,
                reason="texto_vacio",
                confidence=0.0,
                risk_flags=["empty"],
            )

        if hypothesis.vsr_confidence is not None and hypothesis.vsr_confidence < 0.25:
            return CommitDecision(
                action=CommitAction.LOW_CONFIDENCE,
                reason="confianza_vsr_baja",
                confidence=hypothesis.vsr_confidence,
                risk_flags=["low_vsr_confidence"],
            )

        if self._is_repetitive(tokens):
            return CommitDecision(
                action=CommitAction.LOW_CONFIDENCE,
                reason="texto_repetitivo",
                confidence=0.2,
                risk_flags=["repetition"],
            )

        if len(tokens) <= 1:
            return CommitDecision(
                action=CommitAction.LOW_CONFIDENCE,
                reason="muy_poco_contexto",
                confidence=0.2,
                risk_flags=["too_short"],
            )

        lower = " ".join(token.lower() for token in tokens)
        if self._is_dangling(lower, tokens):
            return CommitDecision(
                action=CommitAction.WAIT,
                reason="conector_colgante",
                confidence=0.72,
                risk_flags=["dangling_connector"],
            )

        if text.endswith(self.sentence_endings):
            return CommitDecision(
                action=CommitAction.COMMIT,
                committed_text=text,
                confidence=0.86,
                reason="puntuacion_de_cierre",
                risk_flags=[],
            )

        if len(tokens) >= 8:
            return CommitDecision(
                action=CommitAction.COMMIT,
                committed_text=text,
                confidence=0.66,
                reason="contexto_suficiente_sin_conector_colgante",
                risk_flags=["heuristic_commit"],
            )

        if (hypothesis.ms_since_last_commit or 0) >= 3500 and len(tokens) >= 5:
            return CommitDecision(
                action=CommitAction.COMMIT,
                committed_text=text,
                confidence=0.61,
                reason="timeout_con_contexto_suficiente",
                risk_flags=["timeout_commit"],
            )

        return CommitDecision(
            action=CommitAction.WAIT,
            reason="contexto_insuficiente",
            confidence=0.64,
            risk_flags=["conservative_wait"],
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text)

    def _is_dangling(self, lower_text: str, tokens: list[str]) -> bool:
        if not tokens:
            return True
        if tokens[-1].lower() in self.dangling_tokens:
            return True
        return any(lower_text.endswith(suffix) for suffix in self.dangling_suffixes)

    @staticmethod
    def _is_repetitive(tokens: list[str]) -> bool:
        if len(tokens) < 3:
            return False
        lowered = [token.lower() for token in tokens]
        counts = Counter(lowered)
        if counts.most_common(1)[0][1] / len(lowered) >= 0.6:
            return True
        if len(lowered) > 12:
            return False
        run = 1
        for prev, current in zip(lowered, lowered[1:]):
            run = run + 1 if current == prev else 1
            if run >= 3:
                return True
        return False
