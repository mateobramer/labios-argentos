# Transcript cleaning stronger

Regla principal: si no hay evidencia fuerte, no se reescribe ground truth. Se marca como
candidato de review o se excluye del train si el transcript es malo.

## Niveles

### A. auto_clean_safe

Se autoaplica solo cuando no cambia semantica:

- normalizacion Unicode `NFKC`;
- espacios multiples y `strip`;
- caracteres invisibles/control;
- markers no hablados como tokens completos (`[musica]`, `[aplausos]`, `(musica)`,
  `subtitulos`, `suscribete`);
- entidades locales solo si hay lexicon explicito, distancia chica, `source_hint` en
  `source_id` y no cambia estructura de la oracion.

### B. aggressive_candidate_review

Se detecta pero no se autoaplica:

- token larguisimo o inverosimil;
- caracteres raros;
- rachas consonanticas;
- repeticiones anomalas;
- texto demasiado corto/largo para `n_frames`;
- ratio de basura alto;
- texto vacio/no linguistico.

Estos casos van a `transcript_cleaning_candidates.csv`.

### C. transcript_usability

- `usable`: sin problemas o solo `auto_clean_safe`.
- `questionable`: candidatos agresivos pero no evidencia suficiente para excluir.
- `bad_candidate`: texto vacio, no linguistico, ratio de basura alto, token extremo o
  solo markers no hablados.

Ante duda, `questionable`.

## Lexicon

Archivo versionado:

```text
evaluation/experiments/batch_vsr/entity_lexicon.csv
```

Schema:

```csv
canonical,aliases,source_hint,notes
```

Es chico y editable. No se usa para reescrituras masivas.

## Outputs

Generar con:

```bash
python -m evaluation.src.transcript_cleaning \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr
```

Salidas versionables:

```text
evaluation/outputs/batch_vsr/transcript_cleaning_changes.csv
evaluation/outputs/batch_vsr/transcript_cleaning_candidates.csv
evaluation/outputs/batch_vsr/transcript_quality_policy.csv
evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/train.csv
evaluation/outputs/batch_vsr/splits_transcript_cleaned_stronger/val.csv
evaluation/outputs/batch_vsr/splits_all_combined/train.csv
evaluation/outputs/batch_vsr/splits_all_combined/val.csv
```

Overlays generables e ignorados:

```text
evaluation/outputs/batch_vsr/transcripts_current/
evaluation/outputs/batch_vsr/transcripts_cleaned_stronger/
evaluation/outputs/batch_vsr/transcripts_cleaned_restricted/
```

## Split stronger

`E2_transcript_cleaned_stronger` usa:

- train original;
- transcripts `auto_clean_safe`;
- excluye `transcript_usability == bad_candidate` del train;
- mantiene `questionable`;
- val original completa.

`E4_all_combined` usa:

- `visual_cleaned_conservative`;
- `transcript_cleaned_stronger`;
- `lower_face_resized96`.

Si no hay `bad_candidate`, el efecto esperado sigue siendo bajo.
