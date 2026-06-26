import json
import tempfile
import unittest
from pathlib import Path

from realtime.src.provider_factory import make_closure_provider
from realtime.src.secuencias import (
    evaluate_causal_sequence,
    export_llm_annotation_packet,
    load_clips_from_split,
    load_sequence_ground_truth,
    merge_annotation_with_clips,
)


class TestSecuencias(unittest.TestCase):
    def test_load_ground_truth_demo_and_evaluate(self):
        sequence = load_sequence_ground_truth("realtime/examples/ground_truth_demo.json")
        summary = evaluate_causal_sequence(sequence, make_closure_provider("heuristic"))
        self.assertEqual(summary["clips"], 4)
        self.assertEqual(summary["expected_commits"], 2)
        self.assertIn("early_commits", summary)
        self.assertIn("late_waits", summary)

    def test_export_annotation_packet(self):
        clips = load_clips_from_split("vsr_models/splits/val.csv", limit=3)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "packet.md"
            export_llm_annotation_packet(clips, source_id="demo", output_path=output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("Ground truth oracional", text)
            self.assertIn("commit_after_clip", text)
            self.assertIn("Clips ordenados", text)

    def test_merge_annotation_with_clips(self):
        clips = load_clips_from_split(
            "vsr_models/splits/val.csv",
            source_id="CHARLA SOBRE EL AMOR Y EL DESAMOR",
            limit=2,
        )
        annotation = {
            "source_id": "demo_merge",
            "mode": "causal",
            "sentences": [
                {
                    "sentence_id": "s001",
                    "text": clips[0].text,
                    "start_clip": clips[0].clip_id,
                    "end_clip": clips[0].clip_id,
                    "commit_after_clip": clips[0].clip_id,
                    "notes": "",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            annotation_path = Path(tmp) / "annotation.json"
            output_path = Path(tmp) / "ground_truth.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            merge_annotation_with_clips(clips, annotation_path, output_path=output_path, source_id="demo_merge")
            sequence = load_sequence_ground_truth(output_path)
            self.assertEqual(len(sequence.clips), 2)
            self.assertEqual(sequence.sentences[0].commit_after_clip, clips[0].clip_id)


if __name__ == "__main__":
    unittest.main()
