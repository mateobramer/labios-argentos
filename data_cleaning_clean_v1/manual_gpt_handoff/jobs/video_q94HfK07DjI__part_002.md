# Handoff manual GPT cleaning

Contexto del proyecto: `labios-argentos` prepara un dataset de lectura de labios / reconocimiento visual del habla en espa?ol rioplatense. Este job es solo limpieza textual conservadora de transcripciones ASR. ROI/preprocesamiento visual no es requisito para limpiar texto y no debe influir en la decision textual.

Instrucciones manuales:
- Responder SOLO JSONL estricto, una linea JSON por clip principal (`context_only=false`).
- No incluir markdown, comentarios, explicaciones fuera de JSONL ni bloques ```.
- No editar nombres de `clip_id`.
- No devolver ningun output para clips `context_only=true`; son solo contexto.
- No inventar, idealizar, resumir ni borrar disfluencias reales.
- Mantener registro argentino/informal/voseo cuando aparezca.

---

# GPT cleaning por video completo

Unidad de trabajo: un video_id completo. Devolve solo JSONL estricto, una linea por clip elegible.

## Contexto del video

- video_id: q94HfK07DjI
- dataset_group: argentina/new_discovery
- source_id: nd__q94HfK07DjI
- title: El último reportaje del papa Francisco con Infobae
- channel: Infobae
- source_url: https://www.youtube.com/watch?v=q94HfK07DjI

## Reglas conservadoras

# Prompt transcript_clean_v1

Eres un corrector conservador de transcripciones para un dataset de lectura de
labios en espanol rioplatense.

Objetivo: corregir errores claros de transcripcion manteniendo lo que la persona
realmente dijo.

Reglas:

- Large es la base principal.
- Turbo es evidencia secundaria, no verdad absoluta.
- No reescribir libremente.
- No embellecer.
- No normalizar a espanol formal.
- No borrar muletillas reales: eh, tipo, o sea, nada, digamos.
- No borrar repeticiones reales.
- No completar frases que la persona no dijo.
- No inventar contexto.
- Corregir nombres propios, marcas, lugares, jerga y palabras claramente mal
  transcriptas si hay evidencia.
- Usar contexto del video, vecinos, turbo y YouTube transcript como evidencia.
- YouTube transcript es evidencia auxiliar debil, no ground truth.
- Si no estas seguro, marcar `needs_review`.
- Cada cambio necesita evidencia.

Output: JSONL estricto, un objeto por clip.

Schema:

```json
{
  "clip_id": "...",
  "large_text": "...",
  "turbo_text": "...",
  "clean_text": "...",
  "status": "unchanged|patched|needs_review|bad_candidate",
  "confidence": "high|medium|low",
  "patches": [
    {
      "span_before": "...",
      "span_after": "...",
      "patch_type": "entity_fix|slang_fix|asr_word_fix|punctuation_minimal|other",
      "evidence": ["large", "turbo", "context", "youtube_transcript", "neighbor_clip"]
    }
  ],
  "needs_review_reason": null
}
```

Validar que:

- `clean_text` no este vacio para `unchanged`/`patched`.
- `status=patched` tenga al menos un patch.
- `status=unchanged` mantenga `large_text` salvo puntuacion minima.
- `status=needs_review` no aplique cambio dudoso.
- No haya salida fuera del JSONL.

- ROI no es requisito para limpiar texto.
- Los records con `context_only=true` son solo continuidad de contexto; no devuelvas ninguna salida para esos clip_id.
- Devolve salida solo para clips principales (`context_only=false`).
- No inventar frases ni completar huecos.
- No transformar a espanol idealizado ni borrar disfluencias reales.
- Si large y turbo discrepan mucho, elegir la opcion conservadora o `reject`.
- Si no esta claro, usar `keep` o `reject`.
- Mantener registro argentino/informal/voseo.
- No agregar puntuacion excesiva si cambia sentido.

## Schema de salida

Una linea JSON por clip, sin Markdown:

{"clip_id": "...", "action": "keep | patch | reject", "clean_text": "...", "reason": "...", "confidence": "high | medium | low", "notes": "..."}

Acciones:
- `keep`: conservar el ASR seleccionado; `clean_text` debe ser el texto final conservado.
- `patch`: aplicar una correccion puntual con evidencia; `clean_text` no puede estar vacio.
- `reject`: no aplicar texto limpio para ese clip; no se usara patch.

## Input JSONL

{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0439", "start_time": "2634.0", "end_time": "2640.0", "baseline_text": "", "large_text": "En público una vez no pude reprimirme, que fue por la guerra, estaba diciendo un discurso.", "turbo_text": "En público una vez no pude reprimirme, que fue por la guerra, estaba diciendo un discurso.", "selected_asr_text": "En público una vez no pude reprimirme, que fue por la guerra, estaba diciendo un discurso.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0440", "start_time": "2640.0", "end_time": "2646.0", "baseline_text": "", "large_text": "y ahí me salió, solo no pude reprimirme, pero a escondidas.", "turbo_text": "y ahí me salió eso, no pude reprimirme, pero a escondidas.", "selected_asr_text": "y ahí me salió, solo no pude reprimirme, pero a escondidas.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0441", "start_time": "2646.0", "end_time": "2652.0", "baseline_text": "", "large_text": "que los psiquiatras interpreten. Yo no me interpreto.", "turbo_text": "que los psiquiatras interpreten yo no me interpreto", "selected_asr_text": "que los psiquiatras interpreten. Yo no me interpreto.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0442", "start_time": "2652.0", "end_time": "2658.0", "baseline_text": "", "large_text": "yo a veces el tipo de presión es solo", "turbo_text": "Yo a veces, este tipo de expresiones solo.", "selected_asr_text": "yo a veces el tipo de presión es solo", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0443", "start_time": "2658.0", "end_time": "2664.0", "baseline_text": "", "large_text": "Bueno, y la última, espero llevarme una respuesta.", "turbo_text": "Bueno, y la última, espero llevarme una respuesta.", "selected_asr_text": "Bueno, y la última, espero llevarme una respuesta.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0444", "start_time": "2664.0", "end_time": "2670.0", "baseline_text": "", "large_text": "yo vi la final de fútbol. Yo no estoy viviendo en Argentina, estoy viviendo en los Estados Unidos, pero vi a gente que estaba en la ciudad.", "turbo_text": "yo vi la final de fútbol. Yo no estoy viviendo en Argentina, estoy viviendo en los Estados Unidos, pero vi a Argentina.", "selected_asr_text": "yo vi la final de fútbol. Yo no estoy viviendo en Argentina, estoy viviendo en los Estados Unidos, pero vi a gente que estaba en la ciudad.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0445", "start_time": "2670.0", "end_time": "2676.0", "baseline_text": "", "large_text": "especialmente a la Argentina porque quería ver si ganaba Argentina ese festejo.", "turbo_text": "especialmente a la Argentina porque quería ver si ganaba Argentina ese festejo.", "selected_asr_text": "especialmente a la Argentina porque quería ver si ganaba Argentina ese festejo.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0446", "start_time": "2676.0", "end_time": "2682.0", "baseline_text": "", "large_text": "Y fue un momento de una catarsis maravillosa en una sociedad muy sufrida.", "turbo_text": "y fue un momento de una catarsis maravillosa en una sociedad muy sufrida.", "selected_asr_text": "Y fue un momento de una catarsis maravillosa en una sociedad muy sufrida.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0447", "start_time": "2682.0", "end_time": "2688.0", "baseline_text": "", "large_text": "Cuando hablo con amigos, a veces escucho y mucha gente me pregunta, ¿qué es lo que más te gusta?", "turbo_text": "Cuando hablo con amigos, a veces escucho y mucha gente...", "selected_asr_text": "Cuando hablo con amigos, a veces escucho y mucha gente me pregunta, ¿qué es lo que más te gusta?", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0448", "start_time": "2688.0", "end_time": "2694.0", "baseline_text": "", "large_text": "dice algo similar podría llegar a ocurrir si el Papa Francisco...", "turbo_text": "dice algo similar podría llegar a ocurrir si el Papa Francisco...", "selected_asr_text": "dice algo similar podría llegar a ocurrir si el Papa Francisco...", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0449", "start_time": "2694.0", "end_time": "2700.0", "baseline_text": "", "large_text": "visita la Argentina piensa en eso sueña con eso", "turbo_text": "visita a la Argentina. Piensa en eso, sueña con eso.", "selected_asr_text": "visita la Argentina piensa en eso sueña con eso", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0450", "start_time": "2700.0", "end_time": "2706.0", "baseline_text": "", "large_text": "posibilidad pensé en eso estaba planeado en diciembre del 10", "turbo_text": "posibilidad. Pensé en eso, estaba planeado en diciembre del 17.", "selected_asr_text": "posibilidad pensé en eso estaba planeado en diciembre del 10", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "q94HfK07DjI", "job_id": "video_q94HfK07DjI__part_002", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__q94HfK07DjI", "clip_id": "new_discovery::q94HfK07DjI::clip_0451", "start_time": "2706.0", "end_time": "2712.0", "baseline_text": "", "large_text": "se lleva el primero a Chile Argentina y Uruguay", "turbo_text": "Se lleva el primero a Chile, Argentina y Uruguay.", "selected_asr_text": "se lleva el primero a Chile Argentina y Uruguay", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
