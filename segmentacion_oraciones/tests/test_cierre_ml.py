import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from segmentacion_oraciones.src.cierre_ml import (
    LinearClosureProvider,
    evaluate_provider,
    load_training_examples,
    split_examples,
    train_linear_model,
)
from segmentacion_oraciones.src.entrenar_cierre import run_training_pipeline
from segmentacion_oraciones.src.plan_sintetico import build_variations
from segmentacion_oraciones.src.provider_factory import make_closure_provider


class TestCierreML(unittest.TestCase):
    def test_carga_ground_truth_secuencial(self):
        examples = load_training_examples(["segmentacion_oraciones/examples/ground_truth_demo.json"])
        self.assertEqual(len(examples), 4)
        self.assertEqual(examples[0].expected_action, "wait")
        self.assertEqual(examples[1].expected_action, "commit")
        self.assertIn("yo creo que este modulo", examples[1].partial_text)

    def test_carga_formato_sintetico_con_decisiones(self):
        payload = {
            "source_id": "synthetic_demo",
            "synthetic": True,
            "language": "es-AR",
            "dataset_version": "test",
            "generation_config": {
                "context": "streaming",
                "register": "informal",
                "speaker": "joven",
                "noise_level": "alto",
                "difficulty": "repeticiones",
                "split": "train_synthetic_curated",
            },
            "clips": [
                {
                    "clip_id": "clip_0000",
                    "raw_text": "yo creo que falta",
                    "clean_text": "yo creo que falta",
                    "noise_tags": ["none"],
                },
                {
                    "clip_id": "clip_0001",
                    "raw_text": "cerrar cerrar esta idea",
                    "clean_text": "cerrar esta idea",
                    "noise_tags": ["repetition"],
                },
            ],
            "sentences": [
                {
                    "sentence_id": "sent_0000",
                    "text": "yo creo que falta cerrar esta idea",
                    "start_clip": "clip_0000",
                    "end_clip": "clip_0001",
                    "commit_after_clip": "clip_0001",
                    "confidence": 0.9,
                    "boundary_reason": "idea completa",
                }
            ],
            "clip_decisions": [
                {
                    "clip_id": "clip_0000",
                    "visible_context": "yo creo que falta",
                    "action": "wait",
                    "committed_sentence_id": None,
                    "reason": "incompleta",
                },
                {
                    "clip_id": "clip_0001",
                    "visible_context": "yo creo que falta cerrar esta idea",
                    "action": "commit",
                    "committed_sentence_id": "sent_0000",
                    "reason": "completa",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            examples = load_training_examples([path])
        self.assertEqual(len(examples), 2)
        self.assertTrue(examples[0].synthetic)
        self.assertEqual(examples[1].sentence_id, "sent_0000")
        self.assertEqual(examples[1].metadata["input_split"], "train_synthetic_curated")
        self.assertEqual(examples[1].metadata["noise_level"], "alto")
        self.assertIn("repetition", examples[1].metadata["buffer_noise_tags"])
        self.assertEqual(examples[1].metadata["boundary_offset"], 0)

    def test_carga_zip_y_omite_manifest(self):
        payload = {
            "source_id": "synthetic_zip_demo",
            "synthetic": True,
            "language": "es-AR",
            "generation_config": {"split": "dev_synthetic_hard", "noise_level": "medio"},
            "clips": [
                {"clip_id": "clip_0000", "raw_text": "me parece que falta", "noise_tags": ["none"]},
                {"clip_id": "clip_0001", "raw_text": "cerrar la idea", "noise_tags": ["deletion"]},
            ],
            "sentences": [
                {
                    "sentence_id": "sent_0000",
                    "text": "me parece que falta cerrar la idea",
                    "start_clip": "clip_0000",
                    "end_clip": "clip_0001",
                    "commit_after_clip": "clip_0001",
                }
            ],
            "clip_decisions": [
                {
                    "clip_id": "clip_0000",
                    "visible_context": "me parece que falta",
                    "action": "wait",
                    "committed_sentence_id": None,
                },
                {
                    "clip_id": "clip_0001",
                    "visible_context": "me parece que falta cerrar la idea",
                    "action": "commit",
                    "committed_sentence_id": "sent_0000",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("manifest.json", json.dumps({"dataset": "demo"}))
                zf.writestr("train/example.json", json.dumps(payload))
            examples = load_training_examples([path])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].metadata["input_split"], "dev_synthetic_hard")

    def test_entrena_guarda_y_carga_provider_lineal(self):
        examples = load_training_examples(["segmentacion_oraciones/examples/ground_truth_demo.json"])
        model = train_linear_model(examples, include_heuristic=True, epochs=3, seed=7)
        metrics = evaluate_provider(model, examples)
        self.assertEqual(metrics["count"], len(examples))
        self.assertIn("boundary_error_clips", metrics)
        self.assertIn("breakdowns", metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = LinearClosureProvider.load(path)
            provider = make_closure_provider("linear", model_path=str(path))
        self.assertEqual(loaded.name, model.name)
        self.assertEqual(provider.name, model.name)

    def test_pipeline_completo_escribe_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "training"
            summary = run_training_pipeline(
                ["segmentacion_oraciones/examples/ground_truth_demo.json"],
                output_dir=output,
                seed=11,
            )
            self.assertEqual(summary["examples"], 4)
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "best_config.json").exists())
            self.assertIn("best_name", summary["best"])
            splits = split_examples(load_training_examples(["segmentacion_oraciones/examples/ground_truth_demo.json"]), seed=11)
            self.assertGreaterEqual(len(splits["train"]), 1)

    def test_plan_sintetico_genera_grilla_y_lote(self):
        full = build_variations()
        self.assertEqual(len(full), 5040)
        batch = build_variations(seed=3, max_items=12, shuffle=True)
        self.assertEqual(len(batch), 12)
        self.assertEqual(batch[0]["variation_id"], "var_0001")


if __name__ == "__main__":
    unittest.main()
