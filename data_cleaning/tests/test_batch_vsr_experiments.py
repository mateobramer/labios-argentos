import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.src.build_batch_vsr_experiments import build_configs, write_configs
from evaluation.src.parse_batch_vsr_results import parse_all
from evaluation.src.preprocessing_variant import run_preprocessing_variant
from evaluation.src.transcript_cleaning import build_transcript_overlays, limpiar_restringido
from vsr_models.src.fine_tune import build_arg_parser, transcript_txt_path


class TestBatchVsrExperiments(unittest.TestCase):
    def test_transcript_cleaning_no_modifica_originales_y_genera_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits = base / "splits.csv"
            original = base / "data" / "clips" / "fuente_a" / "clip_0001.txt"
            original.parent.mkdir(parents=True)
            original.write_text("hola   mundo\n", encoding="utf-8")
            self._write_csv(
                splits,
                [
                    {
                        "split": "train",
                        "spk": "f01",
                        "titulo": "fuente_a",
                        "clip": "clip_0001",
                        "n_frames": "10",
                        "texto": "hola   mundo",
                        "npz": "data/processed/lip_rois/fuente_a/clip_0001.npz",
                    }
                ],
            )

            summary = build_transcript_overlays(splits, base / "out", repo_root=base)
            rows = self._read_csv(base / "out" / "transcript_cleaning_changes.csv")

            self.assertEqual(original.read_text(encoding="utf-8"), "hola   mundo\n")
            self.assertEqual(summary["transcripts"], 1)
            self.assertEqual(summary["changed"], 1)
            self.assertEqual(rows[0]["changed"], "true")
            self.assertEqual(rows[0]["change_type"], "space_normalization")
            self.assertIn("split_csv_text", rows[0]["evidence"])

    def test_limpieza_restringida_es_local_y_trazable(self):
        cleaned, changes = limpiar_restringido(" hola\u00a0\u00a0mundo\ufffd ")

        self.assertEqual(cleaned, "hola mundo")
        self.assertIn("unicode_normalization", changes)
        self.assertIn("invalid_character_removed", changes)
        self.assertIn("space_normalization", changes)

    def test_transcripts_root_default_y_custom(self):
        base_args = [
            "--gimeno-repo",
            "gimeno",
            "--vsr-config",
            "config.yaml",
            "--load-vsr",
            "model.pth",
            "--rois-root",
            "rois",
        ]
        default = build_arg_parser().parse_args(base_args)
        custom = build_arg_parser().parse_args(base_args + ["--transcripts-root", "cleaned"])

        self.assertEqual(default.transcripts_root, "")
        self.assertEqual(custom.transcripts_root, "cleaned")
        self.assertEqual(transcript_txt_path("cleaned", "fuente", "clip_0001"), str(Path("cleaned") / "fuente" / "clip_0001.txt"))

    def test_batch_builder_genera_e0_e4_y_variant_queda_after_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "transcripts_cleaned_restricted").mkdir(parents=True)
            configs = build_configs(base)

            self.assertEqual(set(configs), {
                "E0_baseline_original",
                "E1_visual_cleaned",
                "E2_transcript_cleaned",
                "E3_preprocessing_variant",
                "E4_all_combined",
            })
            self.assertEqual(configs["E0_baseline_original"]["status"], "ready")
            self.assertEqual(configs["E2_transcript_cleaned"]["status"], "ready")
            self.assertEqual(configs["E3_preprocessing_variant"]["status"], "ready_after_generation")
            self.assertEqual(configs["E4_all_combined"]["status"], "ready_after_generation")

            summary = write_configs(base)
            self.assertEqual(summary["experiments"]["E0_baseline_original"], "ready")
            self.assertTrue((base / "experiments" / "E4_all_combined" / "experiment_config.json").exists())

    def test_preprocessing_smoke_manifest_si_hay_ok_tiene_shape_96(self):
        manifest = Path("evaluation/outputs/batch_vsr/preprocessing_variant_manifest_smoke.csv")
        if not manifest.exists():
            self.skipTest("no hay manifest de smoke")
        for row in self._read_csv(manifest):
            if row["status"] == "ok":
                self.assertRegex(row["shape"], r"^\d+x96x96$")
                self.assertEqual(row["dtype"], "uint8")

    def test_preprocessing_smoke_bloquea_si_falta_mediapipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            summary = run_preprocessing_variant(output_base=out, max_clips=1)

            self.assertIn(summary["status"], {"blocked", "ok", "partial"})
            self.assertTrue(Path(summary["manifest"]).exists())

    def test_parser_funciona_con_fixture_chico(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            test_split = base / "test.csv"
            self._write_csv(
                test_split,
                [
                    {
                        "split": "test",
                        "source_id": "fuente_a",
                        "titulo": "fuente_a",
                        "clip": "clip_0001",
                        "training_usability": "usable",
                        "policy_moderate": "keep",
                    }
                ],
            )
            exp_dir = base / "experiments" / "E0_baseline_original"
            exp_dir.mkdir(parents=True)
            (exp_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "experiment": "E0_baseline_original",
                        "status": "ready",
                        "train_split": "train.csv",
                        "val_split": "val.csv",
                        "test_split": str(test_split),
                        "rois_root": "rois",
                        "transcripts_root": "",
                        "visual_cleaning": "none",
                        "transcript_variant": "current",
                        "preprocessing_variant": "current",
                        "blocked_reason": "",
                    }
                ),
                encoding="utf-8",
            )
            raw = base / "raw" / "E0_baseline_original"
            raw.mkdir(parents=True)
            (raw / "test.inf").write_text("hola mundo#hola\n", encoding="utf-8")

            summary = parse_all(base)
            rows = self._read_csv(base / "results" / "E0_baseline_original.csv")

            self.assertEqual(summary["experiments"][0]["status"], "parsed")
            self.assertEqual(rows[0]["wer"], "0.500000")
            self.assertEqual(rows[0]["transcript_variant"], "current")

    def test_notebook_07_no_contiene_entrenamiento(self):
        data = json.loads(Path("evaluation/notebooks/07_batch_vsr_experiments.ipynb").read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in data["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertNotIn("vsr_models.src.fine_tune", code)
        self.assertNotIn("subprocess", code)
        self.assertNotIn("!python", code)
        self.assertNotIn("gcloud", code)

    def test_no_hay_npz_batch_commiteables(self):
        npz_files = list(Path("evaluation/outputs/batch_vsr").glob("**/*.npz"))
        self.assertEqual(npz_files, [])
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.npz", gitignore)
        self.assertIn("evaluation/outputs/batch_vsr/rois_lower_face_resized96/", gitignore)

    def _write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path):
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
