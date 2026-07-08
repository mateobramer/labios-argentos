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

- video_id: h3HtBhArO1Q
- dataset_group: argentina/new_discovery
- source_id: nd__h3HtBhArO1Q
- title: El Método Rebord #56 - Andy Chango
- channel: El Método Rebord
- source_url: https://www.youtube.com/watch?v=h3HtBhArO1Q

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

{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1570", "start_time": "9420.0", "end_time": "9426.0", "baseline_text": "", "large_text": "Andy, ¿la pasaste bien? Súper, súper, súper, súper. La verdad que sí.", "turbo_text": "Andy, ¿la pasaste bien? Súper, súper, súper, súper. La verdad que sí.", "selected_asr_text": "Andy, ¿la pasaste bien? Súper, súper, súper, súper. La verdad que sí.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1571", "start_time": "9426.0", "end_time": "9432.0", "baseline_text": "", "large_text": "Espectacular. Y mantuvimos bien a nivel producción, fuiste cuidado, fuiste bien tratado.", "turbo_text": "espectacular y mantuvimos bien a nivel producción fuiste cuidado fuiste bien tratado", "selected_asr_text": "Espectacular. Y mantuvimos bien a nivel producción, fuiste cuidado, fuiste bien tratado.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1572", "start_time": "9432.0", "end_time": "9438.0", "baseline_text": "", "large_text": "que fue todo alucinantemente bien parece un milagro estar tantas horas juntos sin beber", "turbo_text": "que fue todo alucinadamente bien parece un milagro estar tantas horas juntos sin beber", "selected_asr_text": "que fue todo alucinantemente bien parece un milagro estar tantas horas juntos sin beber", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1573", "start_time": "9438.0", "end_time": "9444.0", "baseline_text": "", "large_text": "para mí eso. Bueno, no, no, si me van a faltar el respeto, de verdad.", "turbo_text": "para mí eso. Bueno, no, no, si me van a faltar el respeto, de verdad.", "selected_asr_text": "para mí eso. Bueno, no, no, si me van a faltar el respeto, de verdad.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1574", "start_time": "9444.0", "end_time": "9450.0", "baseline_text": "", "large_text": "No, me extraña, en absoluto. Perfectamente hidratado, que es lo que vos querías estar. Para lo que es mi ligero trastorno de...", "turbo_text": "No, me extraña, en absoluto. Perfectamente hidratado, que es lo que vos querías estar. Para lo que es mi ligero trastorno de la...", "selected_asr_text": "No, me extraña, en absoluto. Perfectamente hidratado, que es lo que vos querías estar. Para lo que es mi ligero trastorno de...", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1575", "start_time": "9450.0", "end_time": "9456.0", "baseline_text": "", "large_text": "Si da a estar casi dos horas sin beber es un milagro que solo lo hice por vos. Te googleé antes, vi que tenía gente...", "turbo_text": "ansiedad, estar casi dos horas sin me ver es un milagro que solo lo hice por vos te googleé antes, vi que tenía gente", "selected_asr_text": "Si da a estar casi dos horas sin beber es un milagro que solo lo hice por vos. Te googleé antes, vi que tenía gente...", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1576", "start_time": "9456.0", "end_time": "9462.0", "baseline_text": "", "large_text": "Tranquilo, dije, Andy, da lo mejor de vos. Perfecto, hermoso. Bueno, Andy, muchísimas gracias por mi parte.", "turbo_text": "Tranquilo, dije Andy, da lo mejor de vos. Perfecto, hermoso. Bueno Andy, muchísimas gracias por mi parte.", "selected_asr_text": "Tranquilo, dije, Andy, da lo mejor de vos. Perfecto, hermoso. Bueno, Andy, muchísimas gracias por mi parte.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1577", "start_time": "9462.0", "end_time": "9468.0", "baseline_text": "", "large_text": "por haber venido y compartir un rato. Me divertí mucho, la pasé bien. Le doy las gracias también, como siempre, a nuestro equipo de televisión.", "turbo_text": "por haber venido y compartir un rato. Me divertí mucho, la pasé bien. Le doy las gracias también, como siempre, a nuestro equipo de...", "selected_asr_text": "por haber venido y compartir un rato. Me divertí mucho, la pasé bien. Le doy las gracias también, como siempre, a nuestro equipo de televisión.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1578", "start_time": "9468.0", "end_time": "9474.0", "baseline_text": "", "large_text": "trabajo ya mencionado antes con criterio y rigor en algunos de los breaks que tuvimos y sin más", "turbo_text": "trabajo, ya mencionado antes con criterio y rigor en algunos de los breaks que tuvimos y sin más", "selected_asr_text": "trabajo ya mencionado antes con criterio y rigor en algunos de los breaks que tuvimos y sin más", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "h3HtBhArO1Q", "job_id": "video_h3HtBhArO1Q__part_008", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__h3HtBhArO1Q", "clip_id": "new_discovery::h3HtBhArO1Q::clip_1579", "start_time": "9474.0", "end_time": "9480.0", "baseline_text": "", "large_text": "preámbulos me despido también por mi parte, ¿no? Ahí cuando Tommy me haga la señal nos retiramos finitamente. ¡Adiós!", "turbo_text": "preámbulos, me despido también por mi parte, ¿no? Ahí cuando Tommy me haga la señal, nos retiramos finitamente. Adiós.", "selected_asr_text": "preámbulos me despido también por mi parte, ¿no? Ahí cuando Tommy me haga la señal nos retiramos finitamente. ¡Adiós!", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
