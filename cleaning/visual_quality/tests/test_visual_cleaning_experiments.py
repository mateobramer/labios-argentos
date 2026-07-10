import csv
import json
import tempfile
import unittest
from pathlib import Path

from vsr.evaluation.src.build_visual_cleaning_manifests import (
    QUALITY_COLUMNS,
    construir_manifests,
)
from vsr.evaluation.src.experiment_metrics import cer, comparar_experimentos, wer
from vsr.src.fine_tune import SPLITS_DIR, build_arg_parser, split_csv_path


class TestVisualCleaningExperiments(unittest.TestCase):
    def test_manifests_originales_conservan_cantidades(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits = base / "splits.csv"
            policy = base / "policy.csv"
            out = base / "out"
            self._write_csv(
                splits,
                [
                    self._split_row("train", "fuente_a", "clip_0001"),
                    self._split_row("train", "fuente_a", "clip_0002"),
                    self._split_row("val", "fuente_b", "clip_0001"),
                    self._split_row("test", "fuente_c", "clip_0001"),
                ],
            )
            self._write_csv(
                policy,
                [
                    self._policy_row("train", "fuente_a", "clip_0001", "usable"),
                    self._policy_row("train", "fuente_a", "clip_0002", "bad_candidate"),
                    self._policy_row("val", "fuente_b", "clip_0001", "bad_candidate"),
                    self._policy_row("test", "fuente_c", "clip_0001", "bad_candidate"),
                ],
            )

            resumen = construir_manifests(splits, policy, out)

            self.assertEqual(resumen["original_train"], 2)
            self.assertEqual(resumen["original_val"], 1)
            self.assertEqual(resumen["original_test"], 1)
            self.assertEqual(resumen["visual_cleaned_train"], 1)
            self.assertEqual(resumen["visual_cleaned_val"], 1)
            self.assertEqual(resumen["visual_cleaned_test_original"], 1)
            self.assertEqual(resumen["excluded_train_bad_candidate"], 1)
            self.assertEqual(resumen["compatible_splits"]["splits_original_train"], 2)
            self.assertEqual(resumen["compatible_splits"]["splits_visual_cleaned_train"], 1)

    def test_cleaned_train_excluye_bad_candidate_y_test_no_filtra(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits = base / "splits.csv"
            policy = base / "policy.csv"
            out = base / "out"
            self._write_csv(
                splits,
                [
                    self._split_row("train", "fuente_a", "clip_0001"),
                    self._split_row("train", "fuente_a", "clip_0002"),
                    self._split_row("test", "fuente_a", "clip_0003"),
                ],
            )
            self._write_csv(
                policy,
                [
                    self._policy_row("train", "fuente_a", "clip_0001", "usable"),
                    self._policy_row("train", "fuente_a", "clip_0002", "bad_candidate"),
                    self._policy_row("test", "fuente_a", "clip_0003", "bad_candidate"),
                ],
            )

            construir_manifests(splits, policy, out)
            train = self._read_csv(out / "visual_cleaned_train.csv")
            test = self._read_csv(out / "visual_cleaned_test_original.csv")
            splits_original_train = self._read_csv(out / "splits_original" / "train.csv")
            splits_visual_cleaned_train = self._read_csv(out / "splits_visual_cleaned" / "train.csv")
            splits_visual_cleaned_val = self._read_csv(out / "splits_visual_cleaned" / "val.csv")

            self.assertEqual([row["clip"] for row in train], ["clip_0001"])
            self.assertEqual([row["clip"] for row in test], ["clip_0003"])
            self.assertEqual([row["clip"] for row in splits_original_train], ["clip_0001", "clip_0002"])
            self.assertEqual([row["clip"] for row in splits_visual_cleaned_train], ["clip_0001"])
            self.assertEqual(splits_visual_cleaned_val, [])
            for col in QUALITY_COLUMNS:
                self.assertIn(col, train[0])

    def test_fine_tune_usa_splits_default_si_no_se_pasa_splits_dir(self):
        args = build_arg_parser().parse_args(
            [
                "--gimeno-repo",
                "gimeno",
                "--vsr-config",
                "config.yaml",
                "--load-vsr",
                "model.pth",
                "--rois-root",
                "rois",
            ]
        )

        self.assertEqual(args.splits_dir, SPLITS_DIR)
        self.assertEqual(split_csv_path(args.splits_dir, "train"), str(Path(SPLITS_DIR) / "train.csv"))

    def test_fine_tune_usa_splits_custom_si_se_pasa_splits_dir(self):
        args = build_arg_parser().parse_args(
            [
                "--gimeno-repo",
                "gimeno",
                "--vsr-config",
                "config.yaml",
                "--load-vsr",
                "model.pth",
                "--rois-root",
                "rois",
                "--splits-dir",
                "vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned",
            ]
        )

        self.assertEqual(args.splits_dir, "vsr/evaluation/outputs/visual_cleaning/splits_visual_cleaned")
        self.assertEqual(split_csv_path(args.splits_dir, "val"), str(Path(args.splits_dir) / "val.csv"))

    def test_builder_no_modifica_splits_canonicos(self):
        train = Path("vsr/splits/train.csv")
        val = Path("vsr/splits/val.csv")
        before_train = train.read_bytes()
        before_val = val.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            construir_manifests(
                Path("vsr/splits/splits.csv"),
                Path("data/metadata/visual_quality_policy_analysis_v2.csv"),
                Path(tmp) / "manifests",
            )

        self.assertEqual(train.read_bytes(), before_train)
        self.assertEqual(val.read_bytes(), before_val)

    def test_wer_cer_en_ejemplos_chicos(self):
        self.assertEqual(wer("hola mundo", "hola mundo"), 0.0)
        self.assertAlmostEqual(wer("hola mundo", "hola"), 0.5)
        self.assertEqual(cer("hola", "hola"), 0.0)
        self.assertAlmostEqual(cer("hola", "ola"), 0.25)

    def test_comparacion_bloquea_grupos_chicos(self):
        rows = [
            {"experiment": "baseline_original", "wer": "0.5", "cer": "0.2"},
            {"experiment": "visual_cleaned", "wer": "0.4", "cer": "0.1"},
        ]

        resumen = comparar_experimentos(rows, min_clips=30)

        self.assertFalse(resumen["can_conclude"])
        self.assertIn("faltan clips", resumen["warning"])

    def test_notebook_no_define_funciones_largas(self):
        notebook = Path("vsr/evaluation/notebooks/06_experimentos_cleaning_vs_original.ipynb")
        data = json.loads(notebook.read_text(encoding="utf-8"))
        code_cells = [cell for cell in data["cells"] if cell.get("cell_type") == "code"]
        self.assertLessEqual(len(data["cells"]), 15)
        self.assertFalse(any("def " in "".join(cell.get("source", [])) for cell in code_cells))

    def test_notebook_no_ejecuta_entrenamiento(self):
        notebook = Path("vsr/evaluation/notebooks/06_experimentos_cleaning_vs_original.ipynb")
        data = json.loads(notebook.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in data["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertNotIn("vsr.src.fine_tune", code)
        self.assertNotIn("subprocess", code)
        self.assertNotIn("get_ipython().system", code)
        self.assertNotIn("!python", code)

    def _split_row(self, split, titulo, clip):
        return {
            "split": split,
            "spk": titulo[-1],
            "titulo": titulo,
            "clip": clip,
            "n_frames": "25",
            "texto": "hola mundo",
            "npz": f"data/processed/lip_rois/{titulo}/{clip}.npz",
        }

    def _policy_row(self, split, source_id, clip, usability):
        return {
            "split": split,
            "spk": source_id[-1],
            "source_id": source_id,
            "titulo": source_id,
            "clip": clip,
            "training_usability": usability,
            "policy_moderate": "exclude" if usability == "bad_candidate" else "keep",
            "review_score": "0.0",
            "quality_score": "0.9",
        }

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
