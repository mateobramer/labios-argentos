import json
import tempfile
import unittest
from pathlib import Path

from data_cleaning.src.llm_text_corrections import generar_revision


class TestLLMTextCorrections(unittest.TestCase):
    def test_generar_revision_detecta_raw_mismatch(self):
        suggestions = {
            "source_id": "CHARLA SOBRE EL AMOR Y EL DESAMOR",
            "corrections": [
                {
                    "clip_id": "clip_0000",
                    "raw_text": "texto que no coincide",
                    "suggested_text": "voy a antereccionar maria del cerro y ahora que hay un monton de gente en el chat les quiero decir algo",
                    "action": "corrected",
                    "confidence": 0.5,
                    "risk_flags": ["posible_alucinacion"],
                    "notes": "",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            suggestions_path = Path(tmp) / "suggestions.json"
            output_dir = Path(tmp) / "out"
            suggestions_path.write_text(json.dumps(suggestions), encoding="utf-8")
            summary = generar_revision(
                split_path="vsr_models/splits/val.csv",
                suggestions_path=suggestions_path,
                output_dir=output_dir,
            )
            self.assertGreater(summary["validation_flags"]["raw_text_no_coincide"], 0)
            self.assertTrue((output_dir / "review_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
