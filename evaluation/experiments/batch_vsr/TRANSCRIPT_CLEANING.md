# Transcript cleaning restringido

Objetivo: preparar una variante segura `transcript_cleaned_restricted` sin reescribir
ground truth y sin tocar los `.txt` originales.

## Reglas

No se usa LLM. No se reescriben frases. No se borran disfluencias. No se idealiza texto
oral. No se inventan palabras. No se corrigen entidades salvo regla explicita con
evidencia fuerte; en esta version no se aplican reglas de entidades.

Cambios permitidos:

- normalizacion Unicode `NFKC`;
- remocion de caracteres de control invalidos;
- normalizacion de espacios multiples;
- `strip` de bordes.

No se aplica lowercasing extra: los splits actuales ya estan normalizados en minusculas
para entrenamiento.

## Outputs

Generados por:

```bash
python -m evaluation.src.transcript_cleaning \
  --splits vsr_models/splits/splits.csv \
  --output-base evaluation/outputs/batch_vsr
```

Salidas:

```text
evaluation/outputs/batch_vsr/transcripts_current/
evaluation/outputs/batch_vsr/transcripts_cleaned_restricted/
evaluation/outputs/batch_vsr/transcript_cleaning_changes.csv
```

`transcripts_current/` y `transcripts_cleaned_restricted/` son overlays generables. No
modifican `data/clips/<titulo>/<clip>.txt`.

## Schema de cambios

```csv
source_id,clip,original_path,cleaned_path,original_text,cleaned_text,changed,change_type,evidence,confidence
```

`confidence` solo es `high` cuando el cambio es puramente normalizacion Unicode,
espacios o caracteres invalidos. Si no hay cambio, queda `none`.

Si el resultado produce muy pocos cambios o ninguno, eso es aceptable: la variante sirve
para probar que una limpieza restringida y trazable no altera contenido oral.
