"""
Corrector LLM de la hipótesis del VSR.

El modelo de lectura de labios produce español "visualmente plausible" pero con errores
(segmentación de palabras, sustituciones): p.ej. ref "para racing la sudamericana era el
piso" -> hyp "para las iras su americana en al pisto". Este módulo pasa esa hipótesis por
un LLM que reconstruye la oración más probable en rioplatense.

OJO — resultado medido (investigación 2026-07, test rioplatense): el corrector NO mejora
el WER. Prompted empeora +1–2 pts, un corrector entrenado empeoró +14 pts, y el techo
teórico rescoreando el N-best es de apenas −6 pts. El cuello es el VSR, no el post-proc.
Este módulo queda en la demo como capa OPCIONAL de legibilidad (salida más gramatical y
presentable en pantalla), no como mejora de exactitud.

Dos backends (CORRECTOR_BACKEND):
- **ollama** (default): LLM local (gratis, offline). Usa el server de Ollama del Mac.
- **claude**: API de Anthropic (mejor calidad, requiere ANTHROPIC_API_KEY + créditos).

Diseño:
- Sin extended thinking / temperatura baja: corregir una frase corta es una transformación
  rápida; no queremos sumar latencia.
- **Graceful**: si el backend no está disponible (server Ollama apagado, o sin API key),
  devuelve la hipótesis cruda y la demo sigue andando. Nunca rompe el server.
- Salida normalizada igual que el dataset (`limpiar`): minúsculas, sin puntuación ni
  acentos salvo la ñ — así el corregido es comparable por WER contra el ground truth.

Variables de entorno:
    CORRECTOR_BACKEND   ollama | claude   (default ollama)
    CORRECTOR_ENABLED   "0" para desactivar
    OLLAMA_HOST         default http://localhost:11434
    OLLAMA_MODEL        default qwen2.5:7b
    ANTHROPIC_API_KEY   credencial (backend claude)
    CORRECTOR_MODEL     modelo Claude (backend claude; default claude-opus-4-8)
"""

import json
import os
import re
import unicodedata
import urllib.request

BACKEND = os.environ.get("CORRECTOR_BACKEND", "ollama").lower()
HABILITADO = os.environ.get("CORRECTOR_ENABLED", "1") != "0"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
CLAUDE_MODEL = os.environ.get("CORRECTOR_MODEL", "claude-opus-4-8")

SYSTEM = (
    "Sos un corrector de la salida de un modelo de lectura de labios (VSR) en español "
    "rioplatense. La entrada es una transcripción visualmente plausible pero con errores "
    "frecuentes de segmentación de palabras y sustituciones por palabras parecidas en el "
    "movimiento de la boca. Tu tarea es reconstruir la oración más probable que la persona "
    "realmente dijo, en español rioplatense natural y gramatical.\n\n"
    "Reglas:\n"
    "- Devolvé SOLO la oración corregida, una sola línea, sin explicaciones.\n"
    "- No agregues información ni inventes datos que no estén implícitos en la entrada.\n"
    "- Si la entrada es ininteligible, devolvé tu mejor reconstrucción razonable.\n"
    "- Mantené un registro coloquial rioplatense (voseo si corresponde).\n\n"
    "Ejemplos:\n"
    "entrada: ke ase mucho tienpo ke no te beia\n"
    "salida: que hace mucho tiempo que no te veia\n"
    "entrada: boi a comdar pan al super\n"
    "salida: voy a comprar pan al super"
)

# Variante N-best: el VSR pasa varias candidatas para la MISMA frase; la palabra correcta
# suele estar en alguna aunque no en la primera. El LLM combina la evidencia.
SYSTEM_NBEST = (
    "Sos un corrector de la salida de un modelo de lectura de labios (VSR) en español "
    "rioplatense. El VSR produjo varias transcripciones CANDIDATAS para UNA misma frase "
    "corta, ordenadas de más a menos probable. Cada candidata tiene errores distintos de "
    "segmentación y de palabras parecidas en el movimiento de la boca, pero la palabra "
    "correcta suele aparecer en alguna de ellas aunque no en la primera. Combiná la "
    "evidencia de todas las candidatas para reconstruir la oración más probable que la "
    "persona realmente dijo, en español rioplatense natural y gramatical.\n\n"
    "Reglas:\n"
    "- Devolvé SOLO la oración reconstruida, una sola línea, sin explicaciones.\n"
    "- Preferí palabras que aparezcan (o sean coherentes) en varias candidatas.\n"
    "- No inventes datos que no estén sugeridos por ninguna candidata.\n"
    "- Registro coloquial rioplatense (voseo si corresponde)."
)


def _normalizar(texto):
    """Misma forma que el dataset (limpiar): minúsculas, sin acentos salvo ñ, sin puntuación."""
    texto = texto.strip().lower().replace("ñ", "\x00")
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    texto = texto.replace("\x00", "ñ")
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


class CorrectorLLM:
    """Corrige hipótesis del VSR con un LLM (local o Claude). Pasa el crudo si no puede."""

    def __init__(self, backend=BACKEND):
        self.backend = backend
        self.activo = False
        self.cliente = None
        if not HABILITADO:
            self.modelo = "off"
            return
        if backend == "claude":
            self.modelo = CLAUDE_MODEL
            try:
                import anthropic
                self.cliente = anthropic.Anthropic()  # resuelve ANTHROPIC_API_KEY del entorno
                self.activo = bool(os.environ.get("ANTHROPIC_API_KEY"))
            except Exception:
                self.activo = False
        else:  # ollama
            self.modelo = f"ollama:{OLLAMA_MODEL}"
            try:
                urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=3).read()
                self.activo = True
            except Exception:
                self.activo = False

    def corregir(self, hipotesis):
        """1-best: hipotesis (str) -> oración corregida (str). Devuelve la cruda si no puede."""
        hipotesis = (hipotesis or "").strip()
        if not self.activo or not hipotesis:
            return hipotesis
        return self._run(SYSTEM, f"entrada: {hipotesis}\nsalida:", fallback=hipotesis)

    def corregir_nbest(self, candidatas):
        """N-best: lista de candidatas (mejor primero) -> oración reconstruida (str).

        Devuelve la 1-best si no puede corregir. La señal extra de las candidatas suele
        ayudar más que reescribir una sola string (la palabra correcta aparece en el lattice).
        """
        candidatas = [c.strip() for c in (candidatas or []) if c and c.strip()]
        if not self.activo or not candidatas:
            return candidatas[0] if candidatas else ""
        listado = "\n".join(f"{i}. {c}" for i, c in enumerate(candidatas, 1))
        return self._run(SYSTEM_NBEST, f"candidatas:\n{listado}\nsalida:", fallback=candidatas[0])

    def _run(self, system, user, fallback):
        try:
            crudo = self._chat_claude(system, user) if self.backend == "claude" else self._chat_ollama(system, user)
            return _normalizar(crudo) or fallback
        except Exception as e:
            print(f"[corrector] fallo ({type(e).__name__}); devuelvo crudo", flush=True)
            return fallback

    def _chat_ollama(self, system, user):
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 80},
        }).encode()
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", body,
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return r["message"]["content"]

    def _chat_claude(self, system, user):
        resp = self.cliente.messages.create(
            model=CLAUDE_MODEL, max_tokens=256, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return next((b.text for b in resp.content if b.type == "text"), "")


_SINGLETON = None


def get_corrector():
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = CorrectorLLM()
    return _SINGLETON
