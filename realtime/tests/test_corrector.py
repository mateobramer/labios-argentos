import unittest

from realtime.src.corrector import IdentityCorrectionProvider
from realtime.src.validation import fallback_correction, validate_correction_result


class TestIdentityCorrectionProvider(unittest.TestCase):
    def test_identity_conserva_texto(self):
        provider = IdentityCorrectionProvider()
        result = provider.correct("vos tenes razon che")
        self.assertEqual(result.corrected_text, "vos tenes razon che")
        self.assertFalse(result.changed)

    def test_fallback_conserva_texto_crudo(self):
        result = fallback_correction("texto crudo")
        self.assertEqual(result.corrected_text, "texto crudo")
        self.assertIn("fallback", result.risk_flags)

    def test_validacion_invalida_conserva_texto_crudo(self):
        result, used_fallback = validate_correction_result({"corrected_text": ""}, "texto crudo")
        self.assertTrue(used_fallback)
        self.assertEqual(result.corrected_text, "texto crudo")


if __name__ == "__main__":
    unittest.main()
