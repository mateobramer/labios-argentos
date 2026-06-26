import tempfile
import unittest
from pathlib import Path

from realtime.src.simular_flujo import run_simulation


class TestSimularFlujo(unittest.TestCase):
    def test_simulador_corre_con_frases_sinteticas(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            summary = run_simulation(
                [
                    "yo creo que",
                    "ayer fuimos a la cancha y estuvo buenisimo.",
                    "bueno bueno bueno",
                ],
                log_path=base / "logs.jsonl",
                feedback_path=base / "feedback.jsonl",
            )
            self.assertEqual(summary["count"], 3)
            self.assertIn("latency_ms", summary)
            self.assertGreaterEqual(summary["actions"]["commit"], 1)
            self.assertTrue((base / "logs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
