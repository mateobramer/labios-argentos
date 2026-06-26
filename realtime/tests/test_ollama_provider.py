import unittest

from realtime.src.contracts import CommitAction, PartialHypothesis
from realtime.src.llm.ollama_provider import OllamaProvider
from realtime.src.llm.prompts import build_closure_prompt, build_correction_prompt
from realtime.src.llm.schemas import CLOSURE_DECISION_SCHEMA, CORRECTION_RESULT_SCHEMA


class TestOllamaProvider(unittest.TestCase):
    def test_schemas_tienen_campos_requeridos(self):
        self.assertIn("action", CLOSURE_DECISION_SCHEMA["required"])
        self.assertIn("corrected_text", CORRECTION_RESULT_SCHEMA["required"])

    def test_prompts_incluyen_reglas_duras(self):
        closure_prompt = build_closure_prompt(PartialHypothesis(partial_text="yo creo que"))
        correction_prompt = build_correction_prompt("vos tenes razon")
        self.assertIn("No inventes palabras", closure_prompt)
        self.assertIn("No agregues informacion nueva", correction_prompt)

    def test_ollama_caido_fallback_cierre_wait(self):
        provider = OllamaProvider(base_url="http://127.0.0.1:1", timeout_s=0.05)
        decision = provider.decide(PartialHypothesis(partial_text="hola como estas"))
        self.assertEqual(decision.action, CommitAction.WAIT)
        self.assertIn("fallback", decision.risk_flags)

    def test_ollama_caido_fallback_correccion_texto_crudo(self):
        provider = OllamaProvider(base_url="http://127.0.0.1:1", timeout_s=0.05)
        result = provider.correct("texto crudo")
        self.assertEqual(result.corrected_text, "texto crudo")
        self.assertIn("fallback", result.risk_flags)


if __name__ == "__main__":
    unittest.main()
