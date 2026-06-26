import unittest

from realtime.src.contracts import CommitAction, CommitDecision
from realtime.src.validation import validate_commit_decision


class TestValidation(unittest.TestCase):
    def test_dict_valido_pasa(self):
        result, used_fallback = validate_commit_decision(
            {
                "action": "commit",
                "committed_text": "hola como estas",
                "confidence": 0.8,
                "reason": "test",
                "risk_flags": [],
            }
        )
        self.assertFalse(used_fallback)
        self.assertEqual(result.action, CommitAction.COMMIT)

    def test_dataclass_valida_pasa(self):
        result, used_fallback = validate_commit_decision(
            CommitDecision(action=CommitAction.WAIT, reason="test", confidence=0.5)
        )
        self.assertFalse(used_fallback)
        self.assertEqual(result.action, CommitAction.WAIT)

    def test_action_invalida_fallback(self):
        result, used_fallback = validate_commit_decision({"action": "bad"})
        self.assertTrue(used_fallback)
        self.assertEqual(result.action, CommitAction.WAIT)

    def test_commit_sin_texto_fallback(self):
        result, used_fallback = validate_commit_decision({"action": "commit", "committed_text": ""})
        self.assertTrue(used_fallback)
        self.assertEqual(result.action, CommitAction.WAIT)


if __name__ == "__main__":
    unittest.main()
