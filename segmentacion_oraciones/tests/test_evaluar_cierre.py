import tempfile
import unittest
from pathlib import Path

from segmentacion_oraciones.src.evaluar_cierre import DEMO_CASES, evaluate_closure, load_cases
from segmentacion_oraciones.src.provider_factory import make_closure_provider


class TestEvaluarCierre(unittest.TestCase):
    def test_eval_demo_calcula_metricas(self):
        provider = make_closure_provider("heuristic")
        summary = evaluate_closure(DEMO_CASES, provider)
        self.assertEqual(summary["count"], len(DEMO_CASES))
        self.assertIn("commit_precision", summary)
        self.assertIn("latency_ms", summary)

    def test_load_cases_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text('{"partial_text":"yo creo que","expected_action":"wait"}\n', encoding="utf-8")
            cases = load_cases(path)
            self.assertEqual(cases[0]["expected_action"], "wait")


if __name__ == "__main__":
    unittest.main()
