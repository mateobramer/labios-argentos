import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from vsr.evaluation.src.build_batch_vsr_experiments import build_configs, write_configs
from vsr.evaluation.src.batch_vsr_notebook import comparar_resultados, resolver_repo_path, tamanos_experimentos
from vsr.evaluation.src.parse_batch_vsr_results import parse_all
from preprocessing.src.preprocessing_variant import _fallback_original_roi, run_preprocessing_variant
from cleaning.visual_quality.src.transcript_alignment_audit import build_alignment_audit
from cleaning.visual_quality.src.transcript_cleaning import (
    build_transcript_overlays,
    cargar_lexicon,
    auto_clean_safe,
    limpiar_restringido,
)
from cleaning.visual_quality.src.transcript_second_pass_asr import run_second_pass_asr
from vsr.src.fine_tune import build_arg_parser, transcript_txt_path


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
            candidates = self._read_csv(base / "out" / "transcript_cleaning_candidates.csv")
            policy = self._read_csv(base / "out" / "transcript_quality_policy.csv")

            self.assertEqual(original.read_text(encoding="utf-8"), "hola   mundo\n")
            self.assertEqual(summary["transcripts"], 1)
            self.assertEqual(summary["changed"], 1)
            self.assertEqual(rows[0]["changed"], "true")
            self.assertEqual(rows[0]["auto_applied"], "true")
            self.assertEqual(rows[0]["change_type"], "space_normalization")
            self.assertIn("espacios", rows[0]["evidence"])
            self.assertEqual(candidates, [])
            self.assertEqual(policy[0]["transcript_usability"], "usable")

    def test_limpieza_restringida_es_local_y_trazable(self):
        cleaned, changes = limpiar_restringido(" hola\u00a0\u00a0mundo\ufffd ")

        self.assertEqual(cleaned, "hola mundo")
        self.assertIn("unicode_normalization", changes)
        self.assertIn("invalid_character_removed", changes)
        self.assertIn("space_normalization", changes)

    def test_candidates_policy_y_split_stronger_excluyen_bad_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits = base / "splits.csv"
            self._write_csv(
                splits,
                [
                    self._split_row("train", "fuente_a", "clip_0001", "hola mundo"),
                    self._split_row("train", "fuente_a", "clip_0002", "!!!!!!!!!"),
                    self._split_row("val", "fuente_a", "clip_0003", "hola hola hola hola"),
                ],
            )

            summary = build_transcript_overlays(splits, base / "out", repo_root=base)
            policy = self._read_csv(base / "out" / "transcript_quality_policy.csv")
            candidates = self._read_csv(base / "out" / "transcript_cleaning_candidates.csv")
            stronger_train = self._read_csv(base / "out" / "splits_transcript_cleaned_stronger" / "train.csv")

            by_clip = {row["clip"]: row for row in policy}
            self.assertEqual(by_clip["clip_0001"]["transcript_usability"], "usable")
            self.assertEqual(by_clip["clip_0002"]["transcript_usability"], "bad_candidate")
            self.assertEqual(by_clip["clip_0002"]["transcript_policy_moderate"], "exclude")
            self.assertGreaterEqual(len(candidates), 1)
            self.assertEqual([row["clip"] for row in stronger_train], ["clip_0001"])
            self.assertEqual(summary["excluded_by_policy_moderate"], 1)

    def test_entity_lexicon_solo_autoaplica_con_evidencia_fuerte(self):
        lexicon = [{"canonical": "river", "aliases": "riber", "source_hint": "RIVER", "type": "brand", "notes": ""}]

        changed, changes, evidence = auto_clean_safe("vamos riber", "RIVER GANO", lexicon)
        unchanged, no_changes, _ = auto_clean_safe("vamos riber", "BOCA GANO", lexicon)

        self.assertEqual(changed, "vamos river")
        self.assertIn("entity_replacement_high_confidence", changes)
        self.assertTrue(any("lexicon+asr2/source" in item for item in evidence))
        self.assertEqual(unchanged, "vamos riber")
        self.assertNotIn("entity_replacement_high_confidence", no_changes)

    def test_entity_replacement_no_reescribe_frase_completa(self):
        lexicon = [
            {
                "canonical": "maria becerra",
                "aliases": "esta frase completa no corresponde",
                "source_hint": "maria becerra",
                "type": "person",
                "notes": "",
            }
        ]

        cleaned, changes, _ = auto_clean_safe(
            "esta frase completa no corresponde",
            "maria becerra entrevista",
            lexicon,
            asr2_text="maria becerra",
        )

        self.assertEqual(cleaned, "esta frase completa no corresponde")
        self.assertNotIn("entity_replacement_high_confidence", changes)

    def test_cleaned_text_conserva_disfluencias(self):
        cleaned, changes, _ = auto_clean_safe("eh eh bueno hola")

        self.assertEqual(cleaned, "eh eh bueno hola")
        self.assertEqual(changes, [])

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
            (base / "transcripts_cleaned_stronger").mkdir(parents=True)
            self._write_csv(
                base / "transcript_quality_policy.csv",
                [
                    {
                        "source_id": "fuente_a",
                        "clip": "clip_0001",
                        "split": "train",
                        "transcript_usability": "usable",
                        "transcript_reasons": "",
                        "transcript_policy_moderate": "keep",
                    }
                ],
            )
            self._write_csv(
                base / "transcript_second_pass_asr.csv",
                [
                    {
                        "source_id": "fuente_a",
                        "clip": "clip_0001",
                        "split": "train",
                        "clip_path": "clip.mp4",
                        "current_text": "hola mundo",
                        "asr2_text": "",
                        "asr2_model": "test",
                        "status": "blocked",
                        "reason": "blocked_missing_asr_dependency",
                        "asr2_runtime_sec": "0.0",
                    }
                ],
            )
            self._write_csv(
                base / "splits_transcript_cleaned_stronger" / "train.csv",
                [self._split_row("train", "fuente_a", "clip_0001", "hola mundo")],
            )
            self._write_csv(
                base / "splits_transcript_cleaned_stronger" / "val.csv",
                [self._split_row("val", "fuente_a", "clip_0002", "hola")],
            )
            self._write_csv(
                base / "splits_all_combined" / "train.csv",
                [self._split_row("train", "fuente_a", "clip_0001", "hola mundo")],
            )
            self._write_csv(
                base / "splits_all_combined" / "val.csv",
                [self._split_row("val", "fuente_a", "clip_0002", "hola")],
            )
            configs = build_configs(base)

            self.assertEqual(set(configs), {
                "E0_baseline_original",
                "E1_visual_cleaned",
                "E2_transcript_cleaned_stronger",
                "E3_preprocessing_variant",
                "E4_all_combined",
            })
            self.assertEqual(configs["E0_baseline_original"]["status"], "ready")
            self.assertEqual(configs["E2_transcript_cleaned_stronger"]["status"], "blocked_missing_asr2")
            self.assertEqual(configs["E2_transcript_cleaned_stronger"]["transcript_variant"], "transcript_cleaned_stronger")
            self.assertEqual(configs["E2_transcript_cleaned_stronger"]["transcript_policy"], "moderate")
            self.assertEqual(configs["E3_preprocessing_variant"]["status"], "ready_after_generation")
            self.assertEqual(configs["E4_all_combined"]["status"], "ready_after_generation")
            self.assertEqual(configs["E4_all_combined"]["transcript_variant"], "current")

            summary = write_configs(base)
            self.assertEqual(summary["experiments"]["E0_baseline_original"], "ready")
            self.assertTrue((base / "experiments" / "E4_all_combined" / "experiment_config.json").exists())

    def test_preprocessing_smoke_manifest_si_hay_ok_tiene_shape_96(self):
        manifest = Path("vsr/evaluation/outputs/batch_vsr/preprocessing_variant_manifest_smoke.csv")
        if not manifest.exists():
            self.skipTest("no hay manifest de smoke")
        for row in self._read_csv(manifest):
            if row["status"] == "ok":
                self.assertRegex(row["shape"], r"^\d+x96x96$")
                self.assertEqual(row["dtype"], "uint8")

    def test_preprocessing_smoke_bloquea_si_falta_mediapipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            summary = run_preprocessing_variant(output_base=out, max_clips=1, preview_max=1)

            self.assertIn(summary["status"], {"blocked", "ok", "partial"})
            self.assertTrue(Path(summary["manifest"]).exists())
            if summary["status"] != "blocked":
                self.assertLessEqual(summary["previews"], 1)

    def test_preprocessing_variant_fallback_original_roi_es_explicito(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            original = base / "original.npz"
            variant = base / "variant" / "clip_0001.npz"
            rois = np.zeros((3, 96, 96), dtype=np.uint8)
            np.savez_compressed(original, rois=rois)

            row = {
                "titulo": "fuente_a",
                "clip": "clip_0001",
                "npz": str(original),
            }
            result = _fallback_original_roi(row, variant, "sin frames variant; detection_ratio=0.790")

            self.assertIsNotNone(result)
            self.assertTrue(variant.exists())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["shape"], "3x96x96")
            self.assertEqual(result["dtype"], "uint8")
            self.assertIn("fallback_original_roi_after_variant_no_frames", result["reason"])

    def test_asr2_missing_dependency_no_rompe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits = base / "splits.csv"
            self._write_csv(splits, [self._split_row("train", "fuente_a", "clip_0001", "hola mundo")])
            with patch("cleaning.visual_quality.src.transcript_second_pass_asr.detectar_backend", return_value=None):
                summary = run_second_pass_asr(splits_path=splits, output_path=base / "asr2.csv")

            rows = self._read_csv(base / "asr2.csv")
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(rows[0]["status"], "blocked")
            self.assertEqual(rows[0]["reason"], "blocked_missing_asr_dependency")

    def test_alignment_audit_calcula_wer_cer_y_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            asr2 = base / "asr2.csv"
            self._write_csv(
                asr2,
                [
                    {
                        "source_id": "fuente_a",
                        "clip": "clip_0001",
                        "split": "train",
                        "clip_path": "clip.mp4",
                        "current_text": "hola mundo",
                        "asr2_text": "chau planeta distinto",
                        "asr2_model": "test",
                        "status": "ok",
                        "reason": "",
                        "asr2_runtime_sec": "0.1",
                    }
                ],
            )

            build_alignment_audit(asr2, base / "disagreement.csv")
            rows = self._read_csv(base / "disagreement.csv")

            self.assertEqual(rows[0]["disagreement_level"], "high")
            self.assertGreater(float(rows[0]["wer_current_vs_asr2"]), 0.9)
            self.assertGreater(float(rows[0]["cer_current_vs_asr2"]), 0.5)

    def test_high_disagreement_produce_candidate_y_bad_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            splits = base / "splits.csv"
            self._write_csv(splits, [self._split_row("train", "fuente_a", "clip_0001", "hola mundo")])
            self._write_csv(
                base / "asr2.csv",
                [
                    {
                        "source_id": "fuente_a",
                        "clip": "clip_0001",
                        "split": "train",
                        "clip_path": "clip.mp4",
                        "current_text": "hola mundo",
                        "asr2_text": "chau planeta distinto",
                        "asr2_model": "test",
                        "status": "ok",
                        "reason": "",
                        "asr2_runtime_sec": "0.1",
                    }
                ],
            )
            build_alignment_audit(base / "asr2.csv", base / "disagreement.csv")

            build_transcript_overlays(
                splits,
                base / "out",
                repo_root=base,
                asr2_path=base / "asr2.csv",
                asr_disagreement_path=base / "disagreement.csv",
            )

            candidates = self._read_csv(base / "out" / "transcript_cleaning_candidates.csv")
            policy = self._read_csv(base / "out" / "transcript_quality_policy.csv")
            self.assertTrue(any(row["candidate_type"] in {"asr_disagreement", "possible_audio_text_mismatch", "possible_misalignment"} for row in candidates))
            self.assertEqual(policy[0]["transcript_usability"], "bad_candidate")

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

    def test_parser_usa_mapeo_subset_si_existe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            test_split = base / "test.csv"
            self._write_csv(
                test_split,
                [
                    {"split": "test", "titulo": "fuente_a", "clip": "clip_0001"},
                    {"split": "test", "titulo": "fuente_b", "clip": "clip_0002"},
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
            (raw / "test.inf").write_text("chau mundo#chau\n", encoding="utf-8")
            self._write_csv(
                raw / "test_mapeo.csv",
                [
                    {
                        "sampleID": "s01_0000",
                        "spk": "s01",
                        "titulo": "fuente_b",
                        "clip": "clip_0002",
                        "texto": "chau mundo",
                    }
                ],
            )

            parse_all(base)
            rows = self._read_csv(base / "results" / "E0_baseline_original.csv")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_id"], "fuente_b")
            self.assertEqual(rows[0]["clip"], "clip_0002")

    def test_notebook_helper_resuelve_paths_absolutos_de_vm(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_csv(base / "vsr" / "evaluation" / "splits" / "train.csv", [self._split_row("train", "fuente_a", "clip_0001", "hola")])
            self._write_csv(base / "vsr" / "evaluation" / "splits" / "val.csv", [self._split_row("val", "fuente_a", "clip_0002", "hola")])
            self._write_csv(base / "vsr" / "evaluation" / "splits" / "test.csv", [self._split_row("test", "fuente_a", "clip_0003", "hola")])
            configs = [
                {
                    "experiment": "E0_baseline_original",
                    "train_split": "/home/bianc/labios-argentos/evaluation/splits/train.csv",
                    "val_split": "/home/bianc/labios-argentos/evaluation/splits/val.csv",
                    "test_split": "/home/bianc/labios-argentos/evaluation/splits/test.csv",
                }
            ]

            self.assertEqual(resolver_repo_path(configs[0]["train_split"], base), base / "vsr" / "evaluation" / "splits" / "train.csv")
            sizes = tamanos_experimentos(pd.DataFrame(configs), repo_root=base)

            self.assertEqual(sizes.iloc[0]["train"], 1)
            self.assertEqual(sizes.iloc[0]["val"], 1)
            self.assertEqual(sizes.iloc[0]["test"], 1)

    def test_notebook_helper_compara_resultados_vs_e0(self):
        results = pd.DataFrame(
            [
                {"experiment": "E0_baseline_original", "status": "parsed", "rows": 2, "wer": 0.5, "cer": 0.4, "output": "e0.csv"},
                {"experiment": "E2_transcript_cleaned_stronger", "status": "parsed", "rows": 2, "wer": 0.4, "cer": 0.3, "output": "e2.csv"},
            ]
        )

        comparison = comparar_resultados(results)
        e2 = comparison[comparison["experiment"].eq("E2_transcript_cleaned_stronger")].iloc[0]

        self.assertEqual(e2["interpretacion"], "mejora_vs_e0")
        self.assertAlmostEqual(e2["delta_wer_vs_e0"], -0.1)
        self.assertEqual(int(e2["rank_wer"]), 1)

    def test_notebook_07_no_contiene_entrenamiento(self):
        data = json.loads(Path("vsr/evaluation/notebooks/07_batch_vsr_experiments.ipynb").read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in data["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertNotIn("vsr.src.fine_tune", code)
        self.assertNotIn("subprocess", code)
        self.assertNotIn("!python", code)
        self.assertNotIn("gcloud", code)

    def test_notebooks_08_09_no_contienen_entrenamiento(self):
        for notebook in [
            Path("cleaning/visual_quality/notebooks/08_transcript_cleaning_review.ipynb"),
            Path("preprocessing/notebooks/09_preprocessing_variant_review.ipynb"),
        ]:
            data = json.loads(notebook.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in data["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertNotIn("vsr.src.fine_tune", code)
            self.assertNotIn("subprocess", code)
            self.assertNotIn("!python", code)
            self.assertNotIn("gcloud", code)

    def test_no_quedan_notebooks_08_09_en_evaluation(self):
        self.assertFalse(Path("vsr/evaluation/notebooks/08_transcript_cleaning_review.ipynb").exists())
        self.assertFalse(Path("vsr/evaluation/notebooks/09_preprocessing_variant_review.ipynb").exists())

    def test_vm_readiness_existe_y_tiene_e0_e4(self):
        text = Path("vsr/evaluation/experiments/batch_vsr/VM_READINESS.md").read_text(encoding="utf-8")
        for name in [
            "E0_baseline_original",
            "E1_visual_cleaned",
            "E2_transcript_audited",
            "E3_preprocessing_variant",
            "E4_all_combined",
        ]:
            self.assertIn(name, text)

    def test_no_hay_npz_batch_commiteables(self):
        npz_files = list(Path("vsr/evaluation/outputs/batch_vsr").glob("**/*.npz"))
        allowed_roots = [
            Path("vsr/evaluation/outputs/batch_vsr/preprocessing_variant_smoke"),
            Path("vsr/evaluation/outputs/batch_vsr/rois_lower_face_resized96"),
        ]
        unexpected = [
            path
            for path in npz_files
            if not any(path.is_relative_to(root) for root in allowed_roots)
        ]
        self.assertEqual(unexpected, [])
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.npz", gitignore)
        self.assertIn("vsr/evaluation/outputs/batch_vsr/rois_lower_face_resized96/", gitignore)
        self.assertIn("vsr/evaluation/outputs/batch_vsr/preprocessing_variant_preview/", gitignore)
        self.assertIn("vsr/evaluation/outputs/batch_vsr/preprocessing_variant_smoke/**/*.npz", gitignore)
        self.assertIn("vsr/evaluation/outputs/batch_vsr/preprocessing_variant_preview/**/*.png", gitignore)

    def _split_row(self, split, titulo, clip, texto):
        return {
            "split": split,
            "spk": titulo[-1],
            "titulo": titulo,
            "clip": clip,
            "n_frames": "25",
            "texto": texto,
            "npz": f"data/processed/lip_rois/{titulo}/{clip}.npz",
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
