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

- video_id: OrwtPwftIi4
- dataset_group: argentina/new_discovery
- source_id: nd__OrwtPwftIi4
- title: 🎤 Curso de Oratoria con Daniel Colombo | 100% Práctico
- channel: Daniel Colombo
- source_url: https://www.youtube.com/watch?v=OrwtPwftIi4

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

{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0856", "start_time": "5136.0", "end_time": "5142.0", "baseline_text": "", "large_text": "a menos que tenga que hablarle al público y ahí sí puede mirar a la cámara.", "turbo_text": "a menos que tenga que hablarle al público y ahí sí puede mirar a la cámara.", "selected_asr_text": "a menos que tenga que hablarle al público y ahí sí puede mirar a la cámara.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0857", "start_time": "5142.0", "end_time": "5148.0", "baseline_text": "", "large_text": "Por último, le presentamos los distintos recursos audiovisuales con los que puede apoyar.", "turbo_text": "Por último, le presentamos los distintos recursos audiovisuales con los que puede apoyar.", "selected_asr_text": "Por último, le presentamos los distintos recursos audiovisuales con los que puede apoyar.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0858", "start_time": "5148.0", "end_time": "5154.0", "baseline_text": "", "large_text": "su discurso. Repasamos el valor del pizarrón, los carteles, el video.", "turbo_text": "su discurso. Repasamos el valor del pizarrón, los carteles, el video.", "selected_asr_text": "su discurso. Repasamos el valor del pizarrón, los carteles, el video.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0859", "start_time": "5154.0", "end_time": "5160.0", "baseline_text": "", "large_text": "El proyector, entre otros, todos con sus ventajas y sus desventajas.", "turbo_text": "El proyector, entre otros, todos con sus ventajas y sus desventajas.", "selected_asr_text": "El proyector, entre otros, todos con sus ventajas y sus desventajas.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0860", "start_time": "5160.0", "end_time": "5166.0", "baseline_text": "", "large_text": "Hemos llegado al final de estas nociones básicas que estoy seguro le ayudaron a la gente.", "turbo_text": "Hemos llegado al final de estas nociones básicas que estoy seguro le ayudaron a hacer un poco más.", "selected_asr_text": "Hemos llegado al final de estas nociones básicas que estoy seguro le ayudaron a la gente.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0861", "start_time": "5166.0", "end_time": "5172.0", "baseline_text": "", "large_text": "al momento de tener que hablar en público. Como sabe, estos DVDs tienen unos bonos.", "turbo_text": "al momento de tener que hablar en público. Como sabe, estos DVDs tienen unos bonus.", "selected_asr_text": "al momento de tener que hablar en público. Como sabe, estos DVDs tienen unos bonos.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0862", "start_time": "5172.0", "end_time": "5178.0", "baseline_text": "", "large_text": "tracks donde podrá encontrar algunos consejos extras sobre temas realmente interesantes por ejemplo", "turbo_text": "tracks donde podrá encontrar algunos consejos extras sobre temas realmente interesantes por ejemplo", "selected_asr_text": "tracks donde podrá encontrar algunos consejos extras sobre temas realmente interesantes por ejemplo", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0863", "start_time": "5178.0", "end_time": "5184.0", "baseline_text": "", "large_text": "cómo diseñar su próximo evento contando con la experiencia de los que saben.", "turbo_text": "cómo diseñar su próximo evento contando con la experiencia de los que saben hay", "selected_asr_text": "cómo diseñar su próximo evento contando con la experiencia de los que saben.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0864", "start_time": "5184.0", "end_time": "5190.0", "baseline_text": "", "large_text": "formatos de auditorios y cada uno comunica situaciones diferentes.", "turbo_text": "formatos de auditorios y cada uno comunica situaciones diferentes.", "selected_asr_text": "formatos de auditorios y cada uno comunica situaciones diferentes.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0865", "start_time": "5190.0", "end_time": "5196.0", "baseline_text": "", "large_text": "será los principios básicos del ceremonial y protocolo. Una experta nos revela algunos", "turbo_text": "serán los principios básicos del ceremonial y protocolo una experta nos revela algún", "selected_asr_text": "será los principios básicos del ceremonial y protocolo. Una experta nos revela algunos", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0866", "start_time": "5196.0", "end_time": "5202.0", "baseline_text": "", "large_text": "unas claves sencillas para recordar y aplicar. ¿Cómo vestirse a la hora de ser un hombre?", "turbo_text": "unas claves sencillas para recordar y aplicar. Cómo vestirse a la hora de ser.", "selected_asr_text": "unas claves sencillas para recordar y aplicar. ¿Cómo vestirse a la hora de ser un hombre?", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0867", "start_time": "5202.0", "end_time": "5208.0", "baseline_text": "", "large_text": "ser oradores. Más tips y consejos de una asesora de imagen profesional con lo que", "turbo_text": "ser oradores. Más tips y consejos de una asesora de imagen profesional con lo que...", "selected_asr_text": "ser oradores. Más tips y consejos de una asesora de imagen profesional con lo que", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0868", "start_time": "5208.0", "end_time": "5214.0", "baseline_text": "", "large_text": "hay que saber para cuidar nuestra imagen pública. También tendremos los consejos de un locutor.", "turbo_text": "hay que saber para cuidar nuestra imagen pública. También tendremos los consejos de un locutor", "selected_asr_text": "hay que saber para cuidar nuestra imagen pública. También tendremos los consejos de un locutor.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0869", "start_time": "5214.0", "end_time": "5220.0", "baseline_text": "", "large_text": "profesional y unos videos públicos que conseguimos en internet como información", "turbo_text": "profesional y unos vídeos públicos que conseguimos en internet como información", "selected_asr_text": "profesional y unos videos públicos que conseguimos en internet como información", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0870", "start_time": "5220.0", "end_time": "5226.0", "baseline_text": "", "large_text": "y complemento del aprendizaje con algunos momentos curiosos de distintos oradores.", "turbo_text": "y complemento del aprendizaje con algunos momentos curiosos de distintos oradores.", "selected_asr_text": "y complemento del aprendizaje con algunos momentos curiosos de distintos oradores.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0871", "start_time": "5226.0", "end_time": "5232.0", "baseline_text": "", "large_text": "Ahora bien, estamos llegando al final y es el momento en que le toca a usted.", "turbo_text": "Ahora bien, estamos llegando al final y es el momento en que le toca a usted.", "selected_asr_text": "Ahora bien, estamos llegando al final y es el momento en que le toca a usted.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0872", "start_time": "5232.0", "end_time": "5238.0", "baseline_text": "", "large_text": "Es el momento de arriesgarse y hablar frente a otros exponiendo sus ideas.", "turbo_text": "Es el momento de arriesgarse y hablar frente a otros exponiendo sus ideas.", "selected_asr_text": "Es el momento de arriesgarse y hablar frente a otros exponiendo sus ideas.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0873", "start_time": "5238.0", "end_time": "5244.0", "baseline_text": "", "large_text": "que hablar en público es como practicar un deporte o un ritmo de baile. Al principio", "turbo_text": "que hablar en público es como practicar un deporte o un ritmo de baile.", "selected_asr_text": "que hablar en público es como practicar un deporte o un ritmo de baile. Al principio", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0874", "start_time": "5244.0", "end_time": "5250.0", "baseline_text": "", "large_text": "exige esfuerzo, disciplina, destreza y sobre todo mucha potencia.", "turbo_text": "exige esfuerzo, disciplina, destreza y sobre todo mucha...", "selected_asr_text": "exige esfuerzo, disciplina, destreza y sobre todo mucha potencia.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0875", "start_time": "5250.0", "end_time": "5256.0", "baseline_text": "", "large_text": "práctica, pero con el correr del tiempo y la experiencia usted podrá gozar de los beneficios.", "turbo_text": "pero con el correr del tiempo y la experiencia usted podrá gozar de los beneficios", "selected_asr_text": "práctica, pero con el correr del tiempo y la experiencia usted podrá gozar de los beneficios.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "OrwtPwftIi4", "job_id": "video_OrwtPwftIi4__part_004", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__OrwtPwftIi4", "clip_id": "new_discovery::OrwtPwftIi4::clip_0876", "start_time": "5256.0", "end_time": "5262.0", "baseline_text": "", "large_text": "en todos los ámbitos de su vida dominando el arte de la oratoria.", "turbo_text": "en todos los ámbitos de su vida, dominando el arte de la oratoria.", "selected_asr_text": "en todos los ámbitos de su vida dominando el arte de la oratoria.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
