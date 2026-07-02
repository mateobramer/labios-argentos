import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_cleaning.src.visual_quality_audit import (
    ENCABEZADO,
    Thresholds,
    decidir,
    guardar_manifest,
    preflight_rois,
    sanity_checks,
    seleccionar_muestra_estratificada,
)
from data_cleaning.src.visual_quality_metrics import VideoFrames, metricas_calidad_frames


ROSTRO_OK = {
    "face_detector": "haar",
    "face_count_score": 1.0,
    "track_stability_score": 1.0,
    "pose_score": 0.9,
    "pose_available": True,
    "multi_face_risk": 0.0,
}


class TestVisualQualityAudit(unittest.TestCase):
    def test_decision_drop_por_hard_fail_roi(self):
        frames = np.full((8, 96, 96), 8, dtype=np.float32)
        video = VideoFrames(frames=frames, n_frames_total=len(frames), fps=25.0, input_kind="roi_npz", path=None)
        calidad = metricas_calidad_frames(video)

        decision = decidir(calidad, ROSTRO_OK, "texto con habla", Thresholds(), input_kind="roi_npz")

        self.assertEqual(decision.decision, "drop")
        self.assertIn("oscuridad_extrema", decision.hard_fail_reasons)
        self.assertIn("freeze_extremo_confirmado", decision.hard_fail_reasons)
        self.assertLess(decision.quality_score, 0.5)

    def test_roi_npz_no_usa_haar_pose_multiface_para_decidir(self):
        calidad = self._calidad_sana()
        rostro = dict(
            ROSTRO_OK,
            face_count_score=0.0,
            track_stability_score=0.0,
            pose_score=0.0,
            multi_face_risk=1.0,
        )

        decision = decidir(
            calidad,
            rostro,
            "texto con habla",
            Thresholds(),
            input_kind="roi_npz",
            speaker_mismatch_risk=1.0,
        )

        self.assertEqual(decision.decision, "keep")
        self.assertNotIn("multiples_caras", decision.used_for_decision_reasons)
        self.assertNotIn("cambio_cara_dominante", decision.used_for_decision_reasons)
        self.assertNotIn("perfil_extremo", decision.used_for_decision_reasons)
        self.assertNotIn("riesgo_boca_texto_o_hablante", decision.used_for_decision_reasons)
        self.assertIn("multiples_caras", decision.invalid_for_input_reasons)
        self.assertIn("pose_proxy_haar", decision.non_applicable_reasons)

    def test_roi_npz_scene_cut_si_puede_generar_review(self):
        calidad = self._calidad_sana()
        calidad["scene_cut_score"] = 0.9
        calidad["scene_cut_count"] = 2
        rostro = dict(ROSTRO_OK, multi_face_risk=0.6)

        decision = decidir(calidad, rostro, "texto con habla", Thresholds(), input_kind="roi_npz")

        self.assertEqual(decision.decision, "review")
        self.assertIn("corte_escena", decision.review_reasons)
        self.assertNotIn("multiples_caras", decision.review_reasons)
        self.assertEqual(decision.hard_fail_reasons, [])

    def test_blur_extremo_solo_genera_review_no_drop(self):
        calidad = self._calidad_sana()
        calidad["blur_score"] = 0.0
        calidad["blur_laplaciano"] = 0.0

        decision = decidir(calidad, ROSTRO_OK, "texto con habla", Thresholds(), input_kind="roi_npz")

        self.assertEqual(decision.decision, "review")
        self.assertIn("blur_extremo", decision.review_reasons)
        self.assertNotIn("blur_extremo", decision.hard_fail_reasons)

    def test_raw_clip_no_dropea_freeze_salvo_extremo(self):
        frames = np.full((12, 96, 96), 120, dtype=np.float32)
        video = VideoFrames(frames=frames, n_frames_total=len(frames), fps=25.0, input_kind="raw_clip", path=None)
        calidad = metricas_calidad_frames(video)

        decision = decidir(calidad, ROSTRO_OK, "texto con habla", Thresholds(), input_kind="raw_clip", audit_confidence="low")

        self.assertEqual(decision.decision, "review")
        self.assertIn("freeze_posible_raw_clip", decision.review_reasons)
        self.assertIn("raw_fallback_low_confidence", decision.review_reasons)
        self.assertNotIn("freeze_extremo_confirmado", decision.hard_fail_reasons)

    def test_manifest_contiene_columnas_esperadas(self):
        row = {col: "" for col in ENCABEZADO}
        row.update({"split": "train", "source_id": "fuente", "clip": "clip_0000", "decision": "keep"})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.csv"
            guardar_manifest([row], output)
            with output.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, ENCABEZADO)
                self.assertIn("run_mode", reader.fieldnames)
                self.assertIn("is_interpretable_for_vsr", reader.fieldnames)
                self.assertIn("interpretation_warning", reader.fieldnames)
                self.assertIn("metric_scope", reader.fieldnames)
                self.assertIn("used_for_decision_reasons", reader.fieldnames)
                self.assertIn("non_applicable_reasons", reader.fieldnames)
                self.assertIn("invalid_for_input_reasons", reader.fieldnames)

    def test_sanity_detecta_reason_no_discriminativa(self):
        rows = [{"decision": "review", "used_for_decision_reasons": "blur"} for _ in range(10)]

        sanity = sanity_checks(rows)

        self.assertEqual(sanity["razones_no_discriminativas"][0]["razon"], "blur")

    def test_sanity_advierte_drop_pct_alto(self):
        rows = [{"decision": "drop", "used_for_decision_reasons": "oscuridad_extrema", "hard_fail_reasons": "oscuridad_extrema"}]
        rows.extend({"decision": "keep", "used_for_decision_reasons": "", "hard_fail_reasons": ""} for _ in range(9))

        sanity = sanity_checks(rows)

        self.assertTrue(any("drop_pct=10.00%" in warning for warning in sanity["warnings"]))

    def test_muestreo_estratificado_respeta_limites(self):
        rows = []
        for split in ["train", "val"]:
            for source in ["a", "b"]:
                for idx in range(5):
                    rows.append({"split": split, "titulo": source, "clip": f"clip_{idx:04d}"})

        sample = seleccionar_muestra_estratificada(rows, clips_per_source=2, max_clips=7, seed=123)

        self.assertLessEqual(len(sample), 7)
        counts = {}
        for row in sample:
            key = (row["split"], row["titulo"])
            counts[key] = counts.get(key, 0) + 1
        self.assertTrue(all(count <= 2 for count in counts.values()))

    def test_cli_corre_con_muestra_chica(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "split.csv"
            for idx in range(2):
                frames = np.full((6, 24, 24), 120 + idx, dtype=np.float32)
                np.savez(Path(tmp) / f"clip_{idx:04d}.npz", rois=frames)
            with split_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["split", "titulo", "clip", "texto", "npz"])
                writer.writeheader()
                for idx in range(2):
                    writer.writerow(
                        {
                            "split": "train",
                            "titulo": "fuente_tmp",
                            "clip": f"clip_{idx:04d}",
                            "texto": "texto con habla",
                            "npz": str(Path(tmp) / f"clip_{idx:04d}.npz"),
                        }
                    )
            output = Path(tmp) / "visual_quality_manifest.csv"
            cmd = [
                sys.executable,
                "-m",
                "data_cleaning.src.visual_quality_audit",
                "--split",
                str(split_path),
                "--output",
                str(output),
                "--max-clips",
                "2",
                "--max-frames",
                "8",
                "--sin-haar",
            ]
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            self.assertTrue(output.exists(), result.stdout + result.stderr)
            with output.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertIn("audit_confidence", rows[0])
            self.assertEqual(rows[0]["run_mode"], "roi_audit")
            self.assertEqual(rows[0]["is_interpretable_for_vsr"], "True")
            self.assertIn("used_for_decision_reasons", rows[0])

    def test_preflight_detecta_rois_ausentes(self):
        rows = [
            {
                "split": "train",
                "titulo": "fuente_inexistente",
                "clip": "clip_0000",
                "npz": "data/processed/lip_rois/fuente_inexistente/clip_0000.npz",
            }
        ]

        preflight = preflight_rois(rows, min_roi_coverage=0.8)

        self.assertEqual(preflight.roi_existing_rows, 0)
        self.assertEqual(preflight.raw_fallback_rows, 0)
        self.assertEqual(preflight.missing_visual_rows, 1)
        self.assertEqual(preflight.run_mode, "mixed_audit")
        self.assertFalse(preflight.is_interpretable_for_vsr)

    def test_require_roi_falla_si_cobertura_baja(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "split.csv"
            with split_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["split", "titulo", "clip", "npz"])
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train",
                        "titulo": "fuente_inexistente",
                        "clip": "clip_0000",
                        "npz": str(Path(tmp) / "no_existe.npz"),
                    }
                )
            cmd = [
                sys.executable,
                "-m",
                "data_cleaning.src.visual_quality_audit",
                "--split",
                str(split_path),
                "--require-roi",
                "--preflight-only",
            ]
            result = subprocess.run(cmd, text=True, capture_output=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ROI coverage below required threshold", result.stderr)

    def test_preflight_roi_audit_si_hay_rois_suficientes(self):
        with tempfile.TemporaryDirectory() as tmp:
            roi_path = Path(tmp) / "clip_0000.npz"
            roi_path.write_bytes(b"placeholder")
            rows = [
                {
                    "split": "train",
                    "titulo": "fuente",
                    "clip": "clip_0000",
                    "npz": str(roi_path),
                }
            ]

            preflight = preflight_rois(rows, min_roi_coverage=0.8)

        self.assertEqual(preflight.roi_existing_rows, 1)
        self.assertEqual(preflight.roi_coverage, 1.0)
        self.assertEqual(preflight.run_mode, "roi_audit")
        self.assertTrue(preflight.is_interpretable_for_vsr)

    def _calidad_sana(self):
        base = np.tile(np.linspace(80, 180, 96, dtype=np.float32), (96, 1))
        frames = []
        for i in range(24):
            frame = base.copy()
            parche = np.indices((46, 62)).sum(axis=0) % 2
            frame[28:74, 17:79] += (parche * 35) + ((i % 6) * 5)
            frames.append(frame)
        video = VideoFrames(
            frames=np.stack(frames, axis=0),
            n_frames_total=len(frames),
            fps=25.0,
            input_kind="roi_npz",
            path=None,
        )
        return metricas_calidad_frames(video)


if __name__ == "__main__":
    unittest.main()
