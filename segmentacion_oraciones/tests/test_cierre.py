import unittest

from segmentacion_oraciones.src.cierre import HeuristicClosureProvider
from segmentacion_oraciones.src.contracts import CommitAction, PartialHypothesis


class TestHeuristicClosureProvider(unittest.TestCase):
    def setUp(self):
        self.provider = HeuristicClosureProvider()

    def test_texto_vacio_low_confidence(self):
        result = self.provider.decide(PartialHypothesis(partial_text=""))
        self.assertEqual(result.action, CommitAction.LOW_CONFIDENCE)
        self.assertEqual(result.committed_text, "")

    def test_frase_incompleta_wait(self):
        result = self.provider.decide(PartialHypothesis(partial_text="me parece que"))
        self.assertEqual(result.action, CommitAction.WAIT)
        self.assertEqual(result.committed_text, "")

    def test_frase_cerrada_commit(self):
        text = "ayer fuimos a la cancha y estuvo buenisimo."
        result = self.provider.decide(PartialHypothesis(partial_text=text))
        self.assertEqual(result.action, CommitAction.COMMIT)
        self.assertEqual(result.committed_text, text)

    def test_repeticion_fuerte_low_confidence(self):
        result = self.provider.decide(PartialHypothesis(partial_text="bueno bueno bueno bueno"))
        self.assertEqual(result.action, CommitAction.LOW_CONFIDENCE)

    def test_repeticion_local_en_buffer_largo_no_bloquea(self):
        text = "yo creo que si no no pasa nada podemos seguir mirando la frase completa"
        result = self.provider.decide(PartialHypothesis(partial_text=text))
        self.assertNotEqual(result.action, CommitAction.LOW_CONFIDENCE)

    def test_conector_colgante_wait(self):
        result = self.provider.decide(PartialHypothesis(partial_text="yo creo que"))
        self.assertEqual(result.action, CommitAction.WAIT)


if __name__ == "__main__":
    unittest.main()
