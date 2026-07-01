import csv
import tempfile
import unittest
from pathlib import Path

from data_cleaning.src.visual_quality_eval_sample import (
    COLUMNAS_MUESTRA,
    escribir_csv,
    generar_muestra_estratificada,
    resumen_muestra,
)


class TestVisualQualityEvalSample(unittest.TestCase):
    def test_muestra_incluye_keep_exclude_y_usability_si_existen(self):
        rows = self._rows_base()

        sample = generar_muestra_estratificada(rows, per_group=3, seed=42)
        summary = resumen_muestra(sample)

        self.assertIn("keep", summary["policy_moderate_v2"])
        self.assertIn("exclude", summary["policy_moderate_v2"])
        self.assertIn("usable", summary["training_usability"])
        self.assertIn("questionable", summary["training_usability"])
        self.assertIn("bad_candidate", summary["training_usability"])

    def test_respeta_seed(self):
        rows = self._rows_base()

        sample_a = generar_muestra_estratificada(rows, per_group=3, seed=123)
        sample_b = generar_muestra_estratificada(rows, per_group=3, seed=123)
        sample_c = generar_muestra_estratificada(rows, per_group=3, seed=456)

        self.assertEqual(sample_a, sample_b)
        self.assertNotEqual([row["clip"] for row in sample_a], [row["clip"] for row in sample_c])

    def test_grupo_chico_incluye_todos_los_ejemplos(self):
        rows = [
            self._row("clip_0001", "exclude", "bad_candidate", "train", "fuente_a"),
            self._row("clip_0002", "exclude", "bad_candidate", "val", "fuente_b"),
        ]

        sample = generar_muestra_estratificada(rows, per_group=100, seed=42)

        self.assertEqual(len(sample), 2)
        self.assertEqual({row["clip"] for row in sample}, {"clip_0001", "clip_0002"})

    def test_balancea_por_split_y_fuente(self):
        rows = self._rows_base()

        sample = generar_muestra_estratificada(rows, per_group=4, seed=42)
        keep_usable = [
            row
            for row in sample
            if row["policy_moderate_v2"] == "keep" and row["training_usability"] == "usable"
        ]

        self.assertGreaterEqual(len({row["split"] for row in keep_usable}), 2)
        self.assertGreaterEqual(len({row["source_id"] for row in keep_usable}), 2)

    def test_csv_contiene_columnas_esperadas(self):
        sample = generar_muestra_estratificada(self._rows_base(), per_group=2, seed=42)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.csv"
            escribir_csv(sample, output)
            with output.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, COLUMNAS_MUESTRA)

    def _rows_base(self):
        rows = []
        for idx in range(8):
            rows.append(self._row(f"clip_u_{idx:04d}", "keep", "usable", "train" if idx % 2 else "val", f"fuente_{idx % 3}"))
        for idx in range(7):
            rows.append(self._row(f"clip_q_{idx:04d}", "keep", "questionable", "test" if idx % 2 else "train", f"fuente_{idx % 2}"))
        for idx in range(5):
            rows.append(self._row(f"clip_b_{idx:04d}", "exclude", "bad_candidate", "val" if idx % 2 else "train", f"fuente_{idx % 2}"))
        return rows

    def _row(self, clip, policy, usability, split, source_id):
        return {
            "source_id": source_id,
            "clip": clip,
            "split": split,
            "path_roi": f"data/processed/lip_rois/{source_id}/{clip}.npz",
            "path_text": f"data/clips/{source_id}/{clip}.txt",
            "policy_moderate": policy,
            "training_usability": usability,
            "review_severity": "none" if usability == "usable" else "high",
            "review_score": "0.0" if usability == "usable" else "3.0",
            "policy_moderate_exclusion_reasons": "training_usability:bad_candidate" if policy == "exclude" else "",
            "quality_score": "0.9",
            "mouth_activity_score": "0.5",
            "mouth_visibility_score": "0.6",
            "scene_cut_score": "0.0",
            "blur_score": "0.8",
        }


if __name__ == "__main__":
    unittest.main()
