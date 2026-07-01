import csv
import json
import tempfile
import unittest
from pathlib import Path

from data_cleaning.src.visual_quality_policy_analysis import (
    POLICY_COLUMNS,
    PolicyConfig,
    agregar_analisis_politicas,
    escribir_csv,
    retencion_por,
    resumen_impacto_wer_cer,
)


class TestVisualQualityPolicyAnalysis(unittest.TestCase):
    def test_review_severity_se_calcula(self):
        rows = self._rows_base()

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))

        severities = {row["clip"]: row["review_severity"] for row in enriched}
        self.assertEqual(severities["clip_0000"], "none")
        self.assertEqual(severities["clip_0001"], "low")
        self.assertEqual(severities["clip_0003"], "high")
        self.assertIn("strong_combo", enriched[3]["review_reason_group"])

    def test_training_usability_se_calcula(self):
        rows = self._rows_base()

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))

        usability = {row["clip"]: row["training_usability"] for row in enriched}
        self.assertEqual(usability["clip_0000"], "usable")
        self.assertEqual(usability["clip_0001"], "questionable")
        self.assertEqual(usability["clip_0003"], "bad_candidate")
        self.assertIn("scene_discontinuity", enriched[3]["training_usability_reasons"])

    def test_policy_conservative_no_excluye_reviews_simples(self):
        rows = self._rows_base()

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))

        simple_review = next(row for row in enriched if row["clip"] == "clip_0001")
        self.assertEqual(simple_review["policy_conservative"], "keep")
        self.assertEqual(simple_review["policy_conservative_exclusion_reasons"], "")

    def test_blur_extremo_solo_no_excluye_moderate_v2(self):
        rows = [self._row("clip_0001", "review", "blur_extremo", 0.50, "train", "fuente_a")]

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))

        self.assertEqual(enriched[0]["review_severity"], "high")
        self.assertEqual(enriched[0]["policy_moderate"], "keep")

    def test_quality_tail_solo_no_excluye_moderate_v2(self):
        rows = [
            self._row(f"clip_{idx:04d}", "keep", "", 0.90 + idx / 100.0, "train", "fuente_a")
            for idx in range(10)
        ]
        rows.append(self._row("clip_9999", "keep", "", 0.10, "train", "fuente_a"))

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))
        tail = next(row for row in enriched if row["clip"] == "clip_9999")

        self.assertEqual(tail["review_severity"], "high")
        self.assertEqual(tail["training_usability"], "questionable")
        self.assertEqual(tail["policy_moderate"], "keep")
        self.assertIn("quality_tail", tail["training_usability_reasons"])

    def test_policy_moderate_excluye_combinaciones_fuertes(self):
        rows = self._rows_base()

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))

        combo = next(row for row in enriched if row["clip"] == "clip_0003")
        self.assertEqual(combo["policy_moderate"], "exclude")
        self.assertIn("training_usability:scene_discontinuity", combo["policy_moderate_exclusion_reasons"])

    def test_policy_strict_es_mas_agresiva_que_moderate(self):
        rows = self._rows_base()

        enriched = agregar_analisis_politicas(rows, config=PolicyConfig(moderate_quality_pct=10, strict_quality_pct=20))

        moderate_exclude = sum(row["policy_moderate"] == "exclude" for row in enriched)
        strict_exclude = sum(row["policy_strict"] == "exclude" for row in enriched)
        self.assertGreaterEqual(strict_exclude, moderate_exclude)

    def test_outputs_de_politicas_tienen_columnas_esperadas(self):
        rows = agregar_analisis_politicas(self._rows_base())
        fieldnames = list(self._rows_base()[0].keys()) + POLICY_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "policy.csv"
            escribir_csv(rows, output, fieldnames=fieldnames)
            with output.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertIn("review_severity", reader.fieldnames)
                self.assertIn("policy_moderate", reader.fieldnames)
                self.assertIn("policy_strict_exclusion_reasons", reader.fieldnames)

    def test_retencion_por_split_y_fuente(self):
        enriched = agregar_analisis_politicas(self._rows_base())

        por_split = retencion_por(enriched, "policy_moderate", "split")
        por_fuente = retencion_por(enriched, "policy_moderate", "source_id")

        self.assertTrue(any(item["split"] == "train" for item in por_split))
        self.assertTrue(any(item["source_id"] == "fuente_a" for item in por_fuente))
        self.assertTrue(all("retention_pct" in item for item in por_split))

    def test_wer_cer_insuficiente_no_concluye_impacto(self):
        rows = agregar_analisis_politicas(self._rows_base())
        preds = [{"source_id": rows[0]["source_id"], "clip": rows[0]["clip"], "wer_clip": 0.4, "cer_clip": 0.2}]

        impact = resumen_impacto_wer_cer(rows, preds, min_clips=30)

        self.assertEqual(impact["clips_matcheados"], 1)
        self.assertFalse(impact["suficiente_para_concluir"])
        self.assertIn("no sacar conclusiones", impact["warning"])

    def test_wer_cer_no_concluye_si_no_hay_excludes_matcheados(self):
        base_rows = [
            self._row(f"clip_{idx:04d}", "keep", "", 0.90, "train", "fuente_a")
            for idx in range(30)
        ]
        rows = agregar_analisis_politicas(base_rows)
        keep_rows = [row for row in rows if row["policy_moderate"] == "keep"]
        preds = [
            {"source_id": row["source_id"], "clip": row["clip"], "wer_clip": 0.4, "cer_clip": 0.2}
            for row in keep_rows[:30]
        ]

        impact = resumen_impacto_wer_cer(rows, preds, min_clips=30)

        self.assertTrue(impact["suficiente_para_concluir"])
        self.assertFalse(impact["policy_moderate_comparable"])
        self.assertIn("no valida policy_moderate", impact["warning"])

    def test_notebook_04_es_presentacion_corta(self):
        path = Path("data_cleaning/notebooks/04_auditoria_visual_offline.ipynb")
        nb = json.loads(path.read_text(encoding="utf-8"))

        self.assertLessEqual(len(nb["cells"]), 12)
        self.assertFalse(self._notebook_define_funciones(path))

    def test_notebook_05_no_define_helpers_largos(self):
        path = Path("data_cleaning/notebooks/05_diagnostico_politicas_visuales.ipynb")

        self.assertTrue(path.exists())
        self.assertFalse(self._notebook_define_funciones(path))

    def _rows_base(self):
        return [
            self._row("clip_0000", "keep", "", 0.95, "train", "fuente_a"),
            self._row("clip_0001", "review", "blur", 0.80, "train", "fuente_a"),
            self._row("clip_0002", "review", "contraste_bajo", 0.75, "val", "fuente_b"),
            self._row("clip_0003", "review", "blur_extremo;corte_escena", 0.50, "val", "fuente_b"),
            self._row("clip_0004", "review", "baja_textura_boca", 0.45, "test", "fuente_c"),
        ]

    def _row(self, clip, decision, review_reasons, quality, split, source_id):
        mouth_activity = "0.55"
        mouth_visibility = "0.55"
        mouth_inactive = "0.10"
        mouth_texture = "0.55"
        scene_cut = "0.0"
        blur = "0.80"
        if "boca_inactiva" in review_reasons:
            mouth_activity = "0.05"
        if "baja_textura_boca" in review_reasons:
            mouth_texture = "0.05"
        if "corte_escena" in review_reasons:
            scene_cut = "1.0"
        if "blur_extremo" in review_reasons:
            blur = "0.0"
        return {
            "split": split,
            "source_id": source_id,
            "clip": clip,
            "decision": decision,
            "quality_score": str(quality),
            "used_for_decision_reasons": review_reasons,
            "review_reasons": review_reasons,
            "hard_fail_reasons": "",
            "mouth_activity_score": mouth_activity,
            "mouth_visibility_score": mouth_visibility,
            "mouth_inactive_frame_ratio": mouth_inactive,
            "mouth_texture_score": mouth_texture,
            "scene_cut_score": scene_cut,
            "blur_score": blur,
        }

    def _notebook_define_funciones(self, path):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if any(line.lstrip().startswith("def ") for line in src.splitlines()):
                return True
        return False


if __name__ == "__main__":
    unittest.main()
