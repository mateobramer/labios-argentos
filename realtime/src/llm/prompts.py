"""Prompts versionados para cierre y correccion."""

from __future__ import annotations

import json

from realtime.src.contracts import PartialHypothesis


CLOSURE_PROMPT_VERSION = "closure_v1_conservative"
CORRECTION_PROMPT_VERSION = "correction_v1_preserve_meaning"


def build_closure_prompt(hypothesis: PartialHypothesis) -> str:
    payload = {
        "partial_text": hypothesis.partial_text,
        "last_tokens": hypothesis.last_tokens,
        "stable_prefix": hypothesis.stable_prefix,
        "vsr_confidence": hypothesis.vsr_confidence,
        "ms_since_last_commit": hypothesis.ms_since_last_commit,
    }
    return (
        "Sos un clasificador de cierre de oracion para lectura de labios en espanol "
        "rioplatense. Recibis texto parcial producido por un VSR: puede venir sin "
        "puntuacion, con repeticiones y con errores.\n\n"
        "Decidi solo una accion:\n"
        "- commit: si el texto ya forma una frase u oracion suficientemente cerrada.\n"
        "- wait: si parece que falta contexto.\n"
        "- low_confidence: si el texto esta demasiado roto, repetido o ambiguo.\n\n"
        "Reglas duras:\n"
        "- No inventes palabras.\n"
        "- No completes ideas.\n"
        "- Si dudas entre commit y wait, elegi wait.\n"
        "- Si action no es commit, committed_text debe ser string vacio.\n"
        "- Devolve solo JSON valido con: action, committed_text, confidence, reason, risk_flags.\n\n"
        f"Entrada JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_correction_prompt(raw_text: str, context_before: str = "", context_after: str = "") -> str:
    payload = {
        "raw_text": raw_text,
        "context_before": context_before,
        "context_after": context_after,
    }
    return (
        "Sos un corrector de texto VSR para espanol rioplatense. Corregi solo errores "
        "plausibles de lectura de labios, ortografia, puntuacion o voseo.\n\n"
        "Reglas duras:\n"
        "- No agregues informacion nueva.\n"
        "- No cambies el significado.\n"
        "- No borres nombres propios ni modismos.\n"
        "- Si el texto es demasiado incierto, devolve el mismo texto y marca low_confidence.\n"
        "- Devolve solo JSON valido con: corrected_text, confidence, changed, edits, risk_flags.\n\n"
        f"Entrada JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
