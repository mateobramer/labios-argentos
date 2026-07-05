import unittest

from data_discovery.src.common import SCORE_FIELDS
from data_discovery.src.score_candidates import (
    MAX_CLIPS_PER_SOURCE,
    aplicar_caps,
    build_rows,
    decidir_final,
    row_desde_candidate,
)


class TestScoreCandidates(unittest.TestCase):
    def test_decision_accept_respeta_thresholds(self):
        decision = decidir_final(
            total_score=86,
            visual_score=82,
            audio_score=76,
            context_score=88,
            expected_accent="argentino/rioplatense_probable",
            visual_decision="accept",
            audio_decision="accept",
            audit_status="ok",
        )

        self.assertEqual(decision, "accept")

    def test_reject_si_audio_bajo(self):
        decision = decidir_final(
            total_score=88,
            visual_score=92,
            audio_score=40,
            context_score=95,
            expected_accent="argentino/rioplatense_probable",
            visual_decision="strong_accept",
            audio_decision="reject",
            audit_status="ok",
        )

        self.assertEqual(decision, "reject")

    def test_row_contiene_schema_de_scores(self):
        candidate = self._candidate()
        audit = self._audit(video_id="abc", clips=600)

        row = row_desde_candidate(candidate, audit, clips_per_minute=12)

        for field in SCORE_FIELDS:
            self.assertIn(field, row)
        self.assertIn(row["decision"], {"strong_accept", "accept"})
        self.assertGreater(float(row["accepted_clips_estimate"]), 0)

    def test_accept_sin_minutos_utiles_pasa_a_revision(self):
        candidate = self._candidate()
        audit = self._audit(video_id="abc", clips=0)

        row = row_desde_candidate(candidate, audit, clips_per_minute=12)

        self.assertEqual(row["decision"], "maybe_review")
        self.assertEqual(row["recommended_use"], "manual_review")
        self.assertEqual(float(row["accepted_clips_estimate"]), 0)
        self.assertIn("sin_minutos_utiles_estimados", row["reasons"])

    def test_caps_de_diversidad_bajan_exceso_a_maybe(self):
        rows = []
        for idx in range(3):
            candidate = self._candidate(video_id=f"v{idx}", channel="misma fuente")
            audit = self._audit(video_id=f"v{idx}", duration_minutes=80)
            rows.append(row_desde_candidate(candidate, audit, clips_per_minute=12))

        capped = aplicar_caps(rows)

        accepted = [r for r in capped if r["decision"] in {"strong_accept", "accept"}]
        backup = [r for r in capped if r["decision"] == "maybe_review"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(backup), 2)
        self.assertTrue(all("source_cap_applied_backup" in r["reasons"] for r in backup))

    def test_build_rows_usa_auditoria_por_video_id(self):
        candidates = [self._candidate(video_id="abc")]
        audits = {"abc": self._audit(video_id="abc", clips=500)}

        rows = build_rows(candidates, audits, clips_per_minute=12)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_id"], "abc")

    def _candidate(self, video_id="abc", channel="canal argentino"):
        return {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": "Podcast argentino completo",
            "channel": channel,
            "video_id": video_id,
            "duration_minutes": "60",
            "width": "1920",
            "height": "1080",
            "fps": "30",
            "source_type": "podcast",
            "expected_accent": "argentino/rioplatense_probable",
        }

    def _audit(self, video_id="abc", clips=500, duration_minutes=60):
        return {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "title": "Podcast argentino completo",
            "channel": "canal argentino",
            "duration_minutes": duration_minutes,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "source_type": "podcast",
            "expected_accent": "argentino/rioplatense_probable",
            "visual_quality_score": 92,
            "audio_quality_score": 86,
            "context_score": 94,
            "visual_decision": "strong_accept",
            "audio_decision": "strong_accept",
            "audit_status": "ok",
            "speech_presence_ratio": 0.8,
            "mouth_visible_ratio": min(1.0, clips / 800),
            "single_speaker_visual_proxy": 0.9,
            "visual_reasons": ["visual_proxy_alto"],
            "audio_reasons": ["audio_proxy_alto"],
            "context_reasons": ["fuente_argentina_probable"],
            "uncertainty": "low",
        }


if __name__ == "__main__":
    unittest.main()
