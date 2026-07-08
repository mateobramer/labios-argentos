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

- video_id: YYIVFA000BI
- dataset_group: argentina/new_discovery
- source_id: nd__YYIVFA000BI
- title: Pepe Mujica con Jorge Fontevecchia (Entrevista Completa)
- channel: Perfil
- source_url: https://www.youtube.com/watch?v=YYIVFA000BI

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

{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1132", "start_time": "6792.0", "end_time": "6798.0", "baseline_text": "", "large_text": "Usted dijo que si tenía 40 años menos, vendría a la Argentina y lucharía desde la Argentina.", "turbo_text": "Usted dijo que si tenía 40 años menos, vendría a la Argentina y lucharía desde la Argentina.", "selected_asr_text": "Usted dijo que si tenía 40 años menos, vendría a la Argentina y lucharía desde la Argentina.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1133", "start_time": "6798.0", "end_time": "6804.0", "baseline_text": "", "large_text": "en los cambios que hay que producir. Ayúdenos, déjenos su mensaje de qué haría.", "turbo_text": "en los cambios que hay que producir. Ayúdenos, déjenos su mensaje de qué haría eso.", "selected_asr_text": "en los cambios que hay que producir. Ayúdenos, déjenos su mensaje de qué haría.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": true, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1134", "start_time": "6804.0", "end_time": "6810.0", "baseline_text": "", "large_text": "con 35 años en 2020 en la Argentina.", "turbo_text": "con 35 años en 2020 en la Argentina.", "selected_asr_text": "con 35 años en 2020 en la Argentina.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1135", "start_time": "6810.0", "end_time": "6816.0", "baseline_text": "", "large_text": "¿A qué iniciaría? ¿Cuál sería su norte? ¿Cuál sería su objetivo? ¿Cuál sería su logro?", "turbo_text": "¿A qué iniciaría? ¿Cuál sería su norte? ¿Cuál sería su objetivo? ¿Cuál sería su logro? ¿A dónde?", "selected_asr_text": "¿A qué iniciaría? ¿Cuál sería su norte? ¿Cuál sería su objetivo? ¿Cuál sería su logro?", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1136", "start_time": "6816.0", "end_time": "6822.0", "baseline_text": "", "large_text": "tendría que llegar. Déjenos su legado en términos de un plan. ¿Cuál sería?", "turbo_text": "tendría que llegar déjenos su legado en términos de un plan cuál sería", "selected_asr_text": "tendría que llegar. Déjenos su legado en términos de un plan. ¿Cuál sería?", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1137", "start_time": "6822.0", "end_time": "6828.0", "baseline_text": "", "large_text": "No me he puesto a pensar en eso, pero lo primero que...", "turbo_text": "No me he puesto a pensar en eso, pero lo primero...", "selected_asr_text": "No me he puesto a pensar en eso, pero lo primero que...", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1138", "start_time": "6828.0", "end_time": "6834.0", "baseline_text": "", "large_text": "Primero que le diría, tomaría mucho mate con los que piensan distinto.", "turbo_text": "lo que le diría tomaría mucho mate con los que piensan distinto", "selected_asr_text": "Primero que le diría, tomaría mucho mate con los que piensan distinto.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1139", "start_time": "6834.0", "end_time": "6840.0", "baseline_text": "", "large_text": "mucho tiempo conversando segundo", "turbo_text": "mucho tiempo conversando. Segundo.", "selected_asr_text": "mucho tiempo conversando segundo", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1140", "start_time": "6840.0", "end_time": "6846.0", "baseline_text": "", "large_text": "Trataría de respetar y de incentivar...", "turbo_text": "trataría de respetar y de incentivar", "selected_asr_text": "Trataría de respetar y de incentivar...", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1141", "start_time": "6846.0", "end_time": "6852.0", "baseline_text": "", "large_text": "en todo lo que...", "turbo_text": "en todo lo que...", "selected_asr_text": "en todo lo que...", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1142", "start_time": "6852.0", "end_time": "6858.0", "baseline_text": "", "large_text": "Lo que puede el mundo del trabajo y el mundo de la ciencia.", "turbo_text": "...lo que pueda del mundo del trabajo y al mundo de la ciencia.", "selected_asr_text": "Lo que puede el mundo del trabajo y el mundo de la ciencia.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1143", "start_time": "6858.0", "end_time": "6864.0", "baseline_text": "", "large_text": "Trataría de acotar el despilfarro.", "turbo_text": "Trataría de acotar el despilfarro.", "selected_asr_text": "Trataría de acotar el despilfarro.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1144", "start_time": "6864.0", "end_time": "6870.0", "baseline_text": "", "large_text": "y gastar mucho más en inversión.", "turbo_text": "y gastar mucho más en inversión.", "selected_asr_text": "y gastar mucho más en inversión.", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1145", "start_time": "6870.0", "end_time": "6876.0", "baseline_text": "", "large_text": "En la cabeza de la gente.", "turbo_text": "en la cabeza de la gente", "selected_asr_text": "En la cabeza de la gente.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1146", "start_time": "6876.0", "end_time": "6882.0", "baseline_text": "", "large_text": "Tal vez menos lujo, menos cromado, menos...", "turbo_text": "Tal vez menos lujo, menos cromado, menos...", "selected_asr_text": "Tal vez menos lujo, menos cromado, menos...", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1147", "start_time": "6882.0", "end_time": "6888.0", "baseline_text": "", "large_text": "Confundir progreso con cosas cromadas nuevas.", "turbo_text": "confundir progreso con cosas cromadas nuevas.", "selected_asr_text": "Confundir progreso con cosas cromadas nuevas.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1148", "start_time": "6888.0", "end_time": "6894.0", "baseline_text": "", "large_text": "y mucho más calificación terrestre.", "turbo_text": "y mucho más calificación", "selected_asr_text": "y mucho más calificación terrestre.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1149", "start_time": "6894.0", "end_time": "6900.0", "baseline_text": "", "large_text": "para la gente joven. Pienso que el mundo que viene...", "turbo_text": "para la gente joven. Pienso que el mundo que viene...", "selected_asr_text": "para la gente joven. Pienso que el mundo que viene...", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1150", "start_time": "6900.0", "end_time": "6906.0", "baseline_text": "", "large_text": "lo que llamábamos proletariado en nuestra época", "turbo_text": "lo que llamábamos proletariado en nuestra época", "selected_asr_text": "lo que llamábamos proletariado en nuestra época", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1151", "start_time": "6906.0", "end_time": "6912.0", "baseline_text": "", "large_text": "Era una gente que vestía más o menos de brin y que usaba gorras de cuero de vasco.", "turbo_text": "una gente que vestía más o menos de brin y que usaba gorras de cuero de vasco.", "selected_asr_text": "Era una gente que vestía más o menos de brin y que usaba gorras de cuero de vasco.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1152", "start_time": "6912.0", "end_time": "6918.0", "baseline_text": "", "large_text": "y el proletariado del futuro son gente de túnica o de escritorio.", "turbo_text": "y el proletariado del futuro son gente de túnica o de escritorio", "selected_asr_text": "y el proletariado del futuro son gente de túnica o de escritorio.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1153", "start_time": "6918.0", "end_time": "6924.0", "baseline_text": "", "large_text": "de capacitación terciaria.", "turbo_text": "de capacitación terciaria", "selected_asr_text": "de capacitación terciaria.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1154", "start_time": "6924.0", "end_time": "6930.0", "baseline_text": "", "large_text": "Y la verdadera batalla está en los universitarios.", "turbo_text": "y la verdadera batalla está en las universidades.", "selected_asr_text": "Y la verdadera batalla está en los universitarios.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1155", "start_time": "6930.0", "end_time": "6936.0", "baseline_text": "", "large_text": "las universidades, porque la sociedad implacablemente que viene es la de...", "turbo_text": "las universidades porque la sociedad implacablemente que viene es la de", "selected_asr_text": "las universidades, porque la sociedad implacablemente que viene es la de...", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1156", "start_time": "6936.0", "end_time": "6942.0", "baseline_text": "", "large_text": "conocimiento el problema para qué y para quién trabaja el conocimiento", "turbo_text": "conocimiento el problema para qué y para quién trabaja el conocimiento", "selected_asr_text": "conocimiento el problema para qué y para quién trabaja el conocimiento", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1157", "start_time": "6942.0", "end_time": "6948.0", "baseline_text": "", "large_text": "Pero bueno. Y Pepe, si usted viniera, estamos en este imaginario, ¿en qué partido...", "turbo_text": "Pero bueno. Y Pepe, si usted viniera, estamos en este imaginario, ¿en qué partido...", "selected_asr_text": "Pero bueno. Y Pepe, si usted viniera, estamos en este imaginario, ¿en qué partido...", "disagreement_flags": [], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1158", "start_time": "6948.0", "end_time": "6954.0", "baseline_text": "", "large_text": "se afiliaría, crearía un partido nuevo o se incorporaría al frente.", "turbo_text": "se afiliaría, crearía un partido nuevo o se incorporaría al frente...", "selected_asr_text": "se afiliaría, crearía un partido nuevo o se incorporaría al frente.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1159", "start_time": "6954.0", "end_time": "6960.0", "baseline_text": "", "large_text": "de todos o junto por el cambio? ¿Cómo sería? El partido de la esperanza.", "turbo_text": "de todos o junto por el cambio ¿cómo sería? El partido de la esperanza", "selected_asr_text": "de todos o junto por el cambio? ¿Cómo sería? El partido de la esperanza.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
{"video_id": "YYIVFA000BI", "job_id": "video_YYIVFA000BI__part_005", "context_only": false, "dataset_group": "argentina/new_discovery", "source_id": "nd__YYIVFA000BI", "clip_id": "new_discovery::YYIVFA000BI::clip_1160", "start_time": "6960.0", "end_time": "6966.0", "baseline_text": "", "large_text": "Haría un partido nuevo. El partido de la esperanza.", "turbo_text": "haría un partido nuevo el partido de la esperanza", "selected_asr_text": "Haría un partido nuevo. El partido de la esperanza.", "disagreement_flags": ["large_turbo_disagree"], "disagreement_wer": "", "disagreement_cer": "", "roi_npz_path": "", "usable_for_training": "false"}
