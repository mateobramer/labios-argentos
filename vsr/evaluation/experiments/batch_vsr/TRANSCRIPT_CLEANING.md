# Transcript cleaning stronger con ASR2

Regla principal: no se reemplaza ground truth sin evidencia fuerte. ASR2 y el audit de
desacuerdo producen evidencia; `transcript_cleaning.py` decide overlays, candidates y
policy conservadora.

## Flujo

```bash
python -m cleaning.visual_quality.src.transcript_second_pass_asr \
  --splits vsr/splits/splits.csv \
  --output vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --model large-v3-turbo

python -m cleaning.visual_quality.src.transcript_alignment_audit \
  --asr2 vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --output vsr/evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv

python -m cleaning.visual_quality.src.transcript_cleaning \
  --splits vsr/splits/splits.csv \
  --output-base vsr/evaluation/outputs/batch_vsr \
  --asr2 vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv \
  --asr-disagreement vsr/evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv \
  --lexicon cleaning/visual_quality/resources/entity_lexicon.csv
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
cleaning/visual_quality/resources/entity_lexicon.csv
```

Schema:

```csv
canonical,aliases,source_hint,type,notes
```

No llenar con entidades inventadas masivas.

## Outputs

Versionables/livianos:

```text
vsr/evaluation/outputs/batch_vsr/transcript_second_pass_asr.csv
vsr/evaluation/outputs/batch_vsr/transcript_asr_disagreement.csv
vsr/evaluation/outputs/batch_vsr/transcript_cleaning_changes.csv
vsr/evaluation/outputs/batch_vsr/transcript_cleaning_candidates.csv
vsr/evaluation/outputs/batch_vsr/transcript_quality_policy.csv
vsr/evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/train.csv
vsr/evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/val.csv
vsr/evaluation/outputs/batch_vsr/splits_all_combined/train.csv
vsr/evaluation/outputs/batch_vsr/splits_all_combined/val.csv
```

Generables/ignorados:

```text
vsr/evaluation/outputs/batch_vsr/transcripts_current/
vsr/evaluation/outputs/batch_vsr/transcripts_cleaned_stronger/
vsr/evaluation/outputs/batch_vsr/transcripts_cleaned_restricted/
```

## Decision E2

- `READY_FOR_VM`: ASR2 disponible y policy produce evidencia util.
- `BLOCKED_MISSING_ASR2`: no hay ASR2 usable; no vender E2 como mejora fuerte.
- `LOW_IMPACT_DO_NOT_PRIORITIZE`: ASR2 existe pero no cambia train de forma relevante.
- `REVIEW_NEEDED`: hay evidencia pero requiere revision manual antes de VM.
