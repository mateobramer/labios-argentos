# Ground truth oracional para cierre

Este documento define el formato para anotar donde termina cada oracion cuando una
fuente esta dividida en clips consecutivos.

El objetivo no es evaluar un clip aislado. El cierre real opera sobre una secuencia:

```text
clip_0000 -> buffer parcial -> wait
clip_0001 -> buffer acumulado -> wait
clip_0002 -> buffer acumulado -> commit
```

## Formato que necesito de un LLM fuerte

Para cada fuente o video, usar JSON valido:

```json
{
  "source_id": "NOMBRE_DE_LA_FUENTE",
  "mode": "causal",
  "sentences": [
    {
      "sentence_id": "s001",
      "text": "Oracion completa con puntuacion final.",
      "start_clip": "clip_0000",
      "end_clip": "clip_0002",
      "commit_after_clip": "clip_0002",
      "notes": ""
    }
  ],
  "notes": "Anotacion asistida por LLM y revisada manualmente."
}
```

Campos:

- `source_id`: mismo nombre de fuente que aparece en `vsr_models/splits/*.csv`.
- `sentence_id`: id estable y ordenado: `s001`, `s002`, etc.
- `text`: oracion completa. Se permite agregar puntuacion, pero no informacion nueva.
- `start_clip`: primer clip que aporta texto a la oracion.
- `end_clip`: ultimo clip cuyo texto pertenece a esa oracion.
- `commit_after_clip`: clip despues del cual un sistema causal ya deberia poder
  commitear la oracion. Normalmente coincide con `end_clip`.
- `notes`: dudas o aclaraciones.

## Prompt sugerido

Copiar el archivo exportado por:

```bash
python -m segmentacion_oraciones.src.secuencias export-annotation \
  --split vsr_models/splits/val.csv \
  --source-id "NOMBRE_EXACTO_DE_LA_FUENTE" \
  --limit 40 \
  --output segmentacion_oraciones/outputs/annotation/NOMBRE_FUENTE.md
```

Y pedir:

```text
Separame estos clips en oraciones completas.

Reglas:
- No agregues informacion nueva.
- No reescribas agresivamente el texto.
- Podes agregar puntuacion minima.
- Si una oracion ocupa varios clips, indicame start_clip y end_clip.
- commit_after_clip debe ser el ultimo clip necesario para saber que termino la oracion.
- Devolve solo JSON valido con el esquema pedido.
```

## Uso en evaluacion

Cuando exista una anotacion JSON, se combina con los clips originales y se evalua:

```bash
python -m segmentacion_oraciones.src.secuencias evaluate \
  --ground-truth segmentacion_oraciones/examples/ground_truth_demo.json
```

Metricas relevantes:

- `early_commits`: commits antes del clip esperado;
- `late_waits`: waits donde ya correspondia commitear;
- `missing_commits`: oraciones esperadas que no fueron commiteadas correctamente;
- `commit_precision` y `commit_recall`;
- latencia p50/p95.

Las metricas de casos sinteticos siguen sirviendo como smoke test, pero no como
resultado cientifico de cierre.
