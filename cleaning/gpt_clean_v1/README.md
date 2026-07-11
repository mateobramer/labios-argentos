# Limpieza clean_v1

Artefactos livianos para preparar la limpieza conservadora de transcripciones del
bucket `labios-argentos-vsr-clean-v1`. `patch_log.csv` (5.2 MB) no está versionado;
recuperarlo con `git show dataset-clean-v1:data_cleaning_clean_v1/patch_log.csv`.
`rejected_patches.jsonl` (evidencia de correcciones rechazadas) sí está.

Este directorio no contiene videos, audios, ROIs, credenciales ni cookies. Los
outputs versionables son manifests, prompts, patch logs y reportes chicos.

Estado de esta corrida:

- Los `.mp4` existentes en `lip_rois/` son ROIs 96x96 sin pista de audio.
- Por eso ASR `turbo` sobre datos existentes queda bloqueado hasta reconstruir
  clips con audio desde videos fuente/URLs.
- `clean_v1` se crea como baseline conservador igual a `large_existing`, marcado
  explicitamente como no-GPT-cleaned.

Comando principal:

```bash
python cleaning/gpt_clean_v1/src/build_clean_v1_baseline.py
```
