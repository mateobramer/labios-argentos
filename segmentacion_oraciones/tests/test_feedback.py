import json
import tempfile
import unittest
from pathlib import Path

from segmentacion_oraciones.src.contracts import FeedbackEvent
from segmentacion_oraciones.src.feedback import FeedbackWriter


class TestFeedbackWriter(unittest.TestCase):
    def test_escribe_jsonl_valido(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.jsonl"
            writer = FeedbackWriter(path)
            writer.write(
                FeedbackEvent(
                    segment_id="seg_1",
                    raw_vsr_text="hola como estas",
                    committed_text="hola como estas",
                    corrected_text="hola como estas",
                )
            )
            rows = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            data = json.loads(rows[0])
            self.assertEqual(data["segment_id"], "seg_1")
            self.assertEqual(data["review_status"], "pending")

    def test_rechaza_evento_sin_campos_minimos(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = FeedbackWriter(Path(tmp) / "feedback.jsonl")
            with self.assertRaises(ValueError):
                writer.write({"raw_vsr_text": "hola"})


if __name__ == "__main__":
    unittest.main()
