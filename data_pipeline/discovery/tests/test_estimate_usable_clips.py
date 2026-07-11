import unittest

from data_pipeline.discovery.src.estimate_usable_clips import estimar_clips_aceptados, estimar_minutos_utiles


class TestEstimateUsableClips(unittest.TestCase):
    def test_estima_minutos_sin_contar_duracion_total(self):
        usable = estimar_minutos_utiles(
            video_duration_minutes=100,
            speech_presence_ratio=0.8,
            mouth_visible_ratio=0.5,
            single_speaker_ratio=0.75,
            visual_accept_ratio=0.9,
        )

        self.assertAlmostEqual(usable, 27.0)

    def test_estima_clips_con_clips_por_minuto(self):
        self.assertEqual(estimar_clips_aceptados(10, 12), 120)


if __name__ == "__main__":
    unittest.main()

