import json
import tempfile
import unittest
from pathlib import Path

from segmentacion_oraciones.src.llm_training import export_sft_jsonl


class TestLLMTrainingExport(unittest.TestCase):
    def test_export_sft_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sft.jsonl"
            export_sft_jsonl(["segmentacion_oraciones/examples/ground_truth_demo.json"], output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["messages"][0]["role"], "system")
        user_payload = json.loads(rows[0]["messages"][1]["content"])
        target_payload = json.loads(rows[0]["messages"][2]["content"])
        self.assertIn("visible_context", user_payload)
        self.assertIn(target_payload["action"], {"wait", "commit", "low_confidence"})


if __name__ == "__main__":
    unittest.main()
