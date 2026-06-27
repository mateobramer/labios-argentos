# Prompt para generar data sintetica de cierre con GPT Pro

## Ejemplo real para darle contexto

Usar este archivo como ejemplo de formato y estilo causal:

```text
realtime/ground_truth/charla_amor_desamor.json
```

Contexto sugerido para describirlo en GPT Pro:

- Video real del dataset, estilo streaming / charla informal.
- Registro muy informal, hablante tipo streamer.
- Clips consecutivos como lectura de labios, sin puntuacion confiable.
- Ya tiene oraciones anotadas con `commit_after_clip`.

Para mostrar ruido real de VSR, usar algunas lineas de:

```text
vsr_models/runs/ft01/eval_finetuned.inf
```

Ese archivo esta en formato `referencia#hipotesis_vsr`. La parte despues de `#` sirve
para mostrarle a GPT Pro como puede sonar una salida VSR ruidosa. No usarla como verdad
oracional sin revisar.

## Prompt base

```text
Quiero generar datos sinteticos para entrenar un modelo de baja latencia que decide
cuando cerrar una oracion en transcripciones parciales de lectura de labios.

Toma como referencia el archivo real que te paso:
- contexto del ejemplo: streaming / charla informal
- los clips son consecutivos
- el sistema ve solo el buffer causal hasta el clip actual
- si una idea queda abierta, debe esperar
- si una oracion queda completa, debe marcar commit despues del ultimo clip necesario

Genera una conversacion natural en espanol rioplatense, oral, no literaria, como si una
persona estuviera hablando en un video. Dividila en clips consecutivos.

Configuracion de esta corrida:
- source_id: synthetic_<contexto>_<variation_id>
- context: <context>
- register: <register>
- speaker: <speaker>
- noise_level: <noise_level>
- difficulty: <difficulty>
- N clips: <recommended_clips>

Reglas:
- Cada clip debe tener entre 4 y 14 palabras aproximadamente.
- Algunas oraciones deben ocupar varios clips.
- Inclui frases incompletas y conectores colgantes.
- No cierres una oracion si queda semanticamente abierta.
- No inventes puntuacion dentro de los clips; los clips son texto crudo.
- El campo `visible_context` debe ser el buffer acumulado desde el ultimo commit.
- `clip_decisions.action` debe ser `wait`, `commit` o `low_confidence`.
- Si `action` no es `commit`, `committed_sentence_id` debe ser null.
- Marca siempre `synthetic=true`.
- No mezcles ejemplos reales con sinteticos dentro del mismo archivo.
- Devolve SOLO JSON valido.

Salida obligatoria:

{
  "source_id": "synthetic_<contexto>_<variation_id>",
  "synthetic": true,
  "language": "es-AR",
  "generation_config": {
    "context": "",
    "register": "",
    "speaker": "",
    "noise_level": "",
    "difficulty": ""
  },
  "clips": [
    {
      "clip_id": "clip_0000",
      "raw_text": ""
    }
  ],
  "sentences": [
    {
      "sentence_id": "sent_0000",
      "text": "",
      "start_clip": "clip_0000",
      "end_clip": "clip_0002",
      "commit_after_clip": "clip_0002",
      "confidence": 0.0,
      "boundary_reason": ""
    }
  ],
  "clip_decisions": [
    {
      "clip_id": "clip_0000",
      "visible_context": "",
      "action": "wait",
      "committed_sentence_id": null,
      "reason": ""
    }
  ]
}
```

## Variaciones

La grilla completa tiene 5040 combinaciones:

- context: streaming, universidad, trabajo, futbol, pareja, politica informal, familia, entrevista, clase, salud
- register: muy informal, informal, neutro, tecnico simple
- speaker: joven, adulto, profesor, streamer, estudiante, profesional
- noise_level: bajo, medio, alto
- difficulty: oraciones largas, muletillas, interrupciones, repeticiones, conectores colgantes, cambio brusco de tema, frases incompletas

Para generar un plan por lotes:

```bash
python -m realtime.src.plan_sintetico --shuffle --seed 13 --max 80 --output realtime/outputs/synthetic_plan/lote_001.jsonl
```

Para generar la grilla completa:

```bash
python -m realtime.src.plan_sintetico --output realtime/outputs/synthetic_plan/full_5040.jsonl
```

Recomendacion: empezar con 80-200 variaciones balanceadas, revisar calidad y recien
despues escalar.

## Criterio de mezcla para entrenamiento

No tratar todos los datos igual:

- `synthetic=true`: sirve para variedad y casos borde.
- ground truth real: sirve para validar si el cierre funciona en el dominio real.
- hipotesis VSR/RSV: sirve para robustez al ruido, pero no debe definir sola la verdad.

Lo ideal es entrenar con input ruidoso cuando exista, pero con etiquetas de corte
producidas desde texto limpio/anotacion GPT Pro o revision humana. El VSR aporta el
tipo de error que vera el sistema en vivo; GPT Pro o humanos aportan el criterio de
oracion completa.
