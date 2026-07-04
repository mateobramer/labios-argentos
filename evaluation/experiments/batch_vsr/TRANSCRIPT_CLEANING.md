# Transcript cleaning stronger con ASR2

Regla principal: no se reemplaza ground truth sin evidencia fuerte. ASR2 y el audit de
desacuerdo producen evidencia; `transcript_cleaning.py` decide overlays, candidates y
policy conservadora.

## Flujo

```bash
python -m data_cleaning.src.transcript_second_pass_asr \
  --splits vsr_models/splits/splits.csv \
  --output evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --model large-v3-turbo

python -m data_cleaning.src.transcript_alignment_audit \
  --asr2 evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --output evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv

python -m data_cleaning.src.transcript_cleaning \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr \
  --asr2 evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --asr-disagreement evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv \
  --lexicon data_cleaning/resources/entity_lexicon.csv
```

## Reglas

- `auto_clean_safe`: Unicode NFKC, espacios, caracteres de control y markers no hablados.
- Reemplazos de entidades/slang: solo spans locales de 1-4 tokens, entrada en lexicon,
  evidencia en ASR2 o fuente/contexto, sin reescribir frases completas ni borrar oralidad.
- `candidates`: registra entity/slang replacements no autoaplicados, desacuerdos ASR,
  posible misalignment, hallucination o audio/text mismatch.
- `bad_candidate`: desacuerdo high o mismatch/hallucination/no-speech de alta confianza.
- `questionable`: desacuerdo medium, entity candidate no resuelto, duracion/texto dudoso o
  multiples candidates.
- Ante duda: `questionable`, no `bad_candidate`.

## Lexicon

Archivo versionado:

```text
data_cleaning/resources/entity_lexicon.csv
```

Schema:

```csv
canonical,aliases,source_hint,type,notes
```

No llenar con entidades inventadas masivas.

## Outputs

Versionables/livianos:

```text
evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv
evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv
evaluation/outputs/batch_vsr/transcript_cleaning_changes.csv
evaluation/outputs/batch_vsr/transcript_cleaning_candidates.csv
evaluation/outputs/batch_vsr/transcript_quality_policy.csv
evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/train.csv
evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/val.csv
evaluation/outputs/batch_vsr/splits_all_combined/train.csv
evaluation/outputs/batch_vsr/splits_all_combined/val.csv
```

Generables/ignorados:

```text
evaluation/outputs/batch_vsr/transcripts_current/
evaluation/outputs/batch_vsr/transcripts_cleaned_stronger/
evaluation/outputs/batch_vsr/transcripts_cleaned_restricted/
```

## Decision E2

- `READY_FOR_VM`: ASR2 disponible y policy produce evidencia util.
- `BLOCKED_MISSING_ASR2`: no hay ASR2 usable; no vender E2 como mejora fuerte.
- `LOW_IMPACT_DO_NOT_PRIORITIZE`: ASR2 existe pero no cambia train de forma relevante.
- `REVIEW_NEEDED`: hay evidencia pero requiere revision manual antes de VM.
