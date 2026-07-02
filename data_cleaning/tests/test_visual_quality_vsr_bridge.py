import csv
import tempfile
import unittest
from pathlib import Path

from data_cleaning.src.export_visual_quality_vsr_scenario import (
    MAPPING_COLUMNS,
    exportar_visual_quality_scenario,
)
from data_cleaning.src.visual_quality_vsr_results import cruzar_predicciones, resumen_resultados


class TestVisualQualityVsrBridge(unittest.TestCase):
    def test_exporta_scenario_y_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = self._sample_csv(base)
            output_base = base / "gimeno_data"
            mapping = base / "mapping.csv"

            summary = exportar_visual_quality_scenario(
                sample,
                output_base=output_base,
                mapping_output=mapping,
                database="Rioplatense",
                scenario="visual-quality-sample",
            )

            split_csv = output_base / "Rioplatense" / "splits" / "visual-quality-sample" / "testRioplatense.csv"
            self.assertEqual(summary["clips"], 2)
            self.assertTrue(split_csv.exists())
            self.assertTrue(mapping.exists())
            split_rows = self._read_csv(split_csv)
            self.assertEqual(list(split_rows[0].keys()), ["sampleID"])
            self.assertTrue((output_base / "Rioplatense" / "ROIs" / "vq01" / "vq01_0000.npz").exists())
            self.assertTrue((output_base / "Rioplatense" / "transcriptions" / "vq01" / "vq01_0000.txt").exists())

    def test_mapping_preserva_source_id_y_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = self._sample_csv(base)
            mapping = base / "mapping.csv"

            exportar_visual_quality_scenario(sample, output_base=base / "out", mapping_output=mapping)
            rows = self._read_csv(mapping)

            self.assertEqual(rows[0]["source_id"], "fuente_a")
            self.assertEqual(rows[0]["clip"], "clip_0001")
            self.assertEqual(rows[0]["policy_moderate"], "keep")
            self.assertIn("review_score", rows[0])
            self.assertEqual(list(rows[0].keys()), MAPPING_COLUMNS)

    def test_detecta_paths_faltantes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = base / "sample.csv"
            self._write_csv(
                sample,
                [
                    {
                        "source_id": "fuente_a",
                        "clip": "clip_0001",
                        "split": "train",
                        "path_roi": str(base / "missing.npz"),
                        "path_text": str(base / "missing.txt"),
                        "policy_moderate_v2": "keep",
                        "training_usability": "usable",
                        "review_score": "0",
                        "quality_score": "1",
                    }
                ],
            )

            with self.assertRaises(FileNotFoundError):
                exportar_visual_quality_scenario(sample, output_base=base / "out", mapping_output=base / "mapping.csv")

    def test_rechaza_csv_sin_columnas_obligatorias(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "sample.csv"
            self._write_csv(sample, [{"source_id": "fuente_a", "clip": "clip_0001"}])

            with self.assertRaises(ValueError):
                exportar_visual_quality_scenario(sample, output_base=Path(tmp) / "out", mapping_output=Path(tmp) / "mapping.csv")

    def test_parser_resultados_calcula_wer_cer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mapping = base / "mapping.csv"
            inf = base / "test.inf"
            output = base / "results.csv"
            self._write_csv(
                mapping,
                [
                    {
                        "sample_id": "vq01_0000",
                        "source_id": "fuente_a",
                        "clip": "clip_0001",
                        "split": "train",
                        "scenario_split": "test",
                        "path_roi": "a.npz",
                        "path_text": "a.txt",
                        "policy_moderate": "keep",
                        "training_usability": "usable",
                        "review_score": "0",
                        "quality_score": "1",
                    },
                    {
                        "sample_id": "vq01_0001",
                        "source_id": "fuente_a",
                        "clip": "clip_0002",
                        "split": "train",
                        "scenario_split": "test",
                        "path_roi": "b.npz",
                        "path_text": "b.txt",
                        "policy_moderate": "exclude",
                        "training_usability": "bad_candidate",
                        "review_score": "10",
                        "quality_score": "0.4",
                    },
                ],
            )
            inf.write_text("hola mundo#hola\nab#abc\n", encoding="utf-8")

            rows = cruzar_predicciones(inf, mapping, output_path=output)
            summary = resumen_resultados(rows)

            self.assertEqual(rows[0]["source_id"], "fuente_a")
            self.assertEqual(rows[0]["wer"], 0.5)
            self.assertEqual(rows[1]["cer"], 0.5)
            self.assertTrue(output.exists())
            self.assertEqual(summary["por_policy_moderate"][0]["policy_moderate"], "exclude")
            self.assertTrue(summary["warnings"])

    def _sample_csv(self, base):
        roi_a = base / "roi_a.npz"
        roi_b = base / "roi_b.npz"
        text_a = base / "a.txt"
        text_b = base / "b.txt"
        roi_a.write_bytes(b"npz-a")
        roi_b.write_bytes(b"npz-b")
        text_a.write_text("hola mundo\n", encoding="utf-8")
        text_b.write_text("otra prueba\n", encoding="utf-8")
        sample = base / "sample.csv"
        self._write_csv(
            sample,
            [
                {
                    "source_id": "fuente_a",
                    "clip": "clip_0001",
                    "split": "train",
                    "path_roi": str(roi_a),
                    "path_text": str(text_a),
                    "policy_moderate_v2": "keep",
                    "training_usability": "usable",
                    "review_score": "0.0",
                    "quality_score": "1.0",
                },
                {
                    "source_id": "fuente_b",
                    "clip": "clip_0002",
                    "split": "val",
                    "path_roi": str(roi_b),
                    "path_text": str(text_b),
                    "policy_moderate_v2": "exclude",
                    "training_usability": "bad_candidate",
                    "review_score": "12.0",
                    "quality_score": "0.4",
                },
            ],
        )
        return sample

    def _write_csv(self, path, rows):
        fieldnames = list(rows[0].keys())
        with Path(path).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path):
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))


if __name__ == "__main__":
    unittest.main()
