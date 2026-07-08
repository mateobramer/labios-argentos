# Handoff manual GPT cleaning

Esta carpeta contiene los prompts manuales para limpiar texto con ChatGPT, sin API y sin automatizacion.

## Como usar

1. Abrir cada archivo de `data_cleaning_clean_v1/manual_gpt_handoff/jobs/`.
2. Pegar el contenido completo en ChatGPT con reasoning/thinking alto.
3. Pedir que responda SOLO JSONL estricto, sin Markdown ni explicaciones.
4. Guardar la respuesta exacta en:

   `data_cleaning_clean_v1/raw_outputs/<job_id>_raw.jsonl`

5. No editar nombres de `clip_id`.
6. No mezclar jobs: cada respuesta va al archivo `<job_id>_raw.jsonl` correspondiente.
7. Si GPT responde incompleto, con Markdown o con JSON invalido, repetir ese job o dejarlo pendiente.
8. No devolver outputs para clips `context_only=true`; son solo contexto.

`job_index.csv` lista cada `job_id`, el prompt `.md`, la salida esperada y la cantidad de clips principales/contexto.

Los outputs crudos/validados previos a este handoff quedaron archivados en `data_cleaning_clean_v1/manual_gpt_handoff/archive_pre_manual/` para que `raw_outputs/` no mezcle respuestas viejas con las respuestas manuales nuevas.

## Comandos cuando esten los raw outputs

Validar todos los outputs manuales disponibles:

```powershell
python data_cleaning_clean_v1/src/validate_gpt_video_jsonl.py --all
```

Aplicar solo JSONL validado:

```powershell
python data_cleaning_clean_v1/src/apply_gpt_video_patches.py --all
```

Reconstruir manifests finales:

```powershell
python data_release/scripts/build_full_clean_release_outputs.py
```

Wrappers equivalentes:

```powershell
python data_cleaning_clean_v1/src/validate_all_manual_gpt_outputs.py
python data_cleaning_clean_v1/src/apply_all_manual_gpt_outputs.py
```

## Outputs generados por validacion

- `data_cleaning_clean_v1/validated/<job_id>_validated.jsonl`
- `data_cleaning_clean_v1/validated/<job_id>_rejected.jsonl`
- `data_cleaning_clean_v1/manual_gpt_handoff/reports/manual_gpt_validation_report.csv`

## Outputs generados por aplicacion

- `data_release/clean_gpt_manifest.csv`
- `data_release/final_release_manifest.csv`
- `data_release/final_train_manifest_clean_gpt_v1.csv`
- `data_release/final_eval_manifest_clean_gpt_v1.csv`
- `data_cleaning_clean_v1/patch_log.csv`
- `data_cleaning_clean_v1/rejected_patches.jsonl`
- `data_release/reports/gpt_cleaning_report.md`

## Recordatorio de calidad

La limpieza GPT es conservadora: no inventar frases, no idealizar el registro, no borrar disfluencias reales y no cambiar el sentido. Si large/turbo discrepan mucho o no hay evidencia fuerte, usar `keep` o `reject`.
