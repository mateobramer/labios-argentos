import json
import tempfile
import unittest
from pathlib import Path

from segmentacion_oraciones.src.dataset_cierre import build_cases_from_text, write_jsonl


class TestDatasetCierre(unittest.TestCase):
    def test_build_cases_from_text(self):
        cases = build_cases_from_text(
            "vos tenes razon che gracias por avisarme hoy",
            segment_id="seg",
            source="test",
        )
        expected = {case["expected_action"] for case in cases}
        self.assertIn("wait", expected)
        self.assertIn("commit", expected)

    def test_write_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            write_jsonl(
                [
                    {
                        "partial_text": "yo creo que",
                        "expected_action": "wait",
                        "segment_id": "seg",
                        "source": "test",
                        "case": "demo",
                    }
                ],
                path,
            )
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["expected_action"], "wait")


if __name__ == "__main__":
    unittest.main()
