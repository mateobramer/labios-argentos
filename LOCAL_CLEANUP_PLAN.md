# Local cleanup ejecutado

Fecha: 2026-07-09
Repo: `C:\Users\bianc\NLP Natural Language Processing\labios-argentos`
Branch: `feature/full-clean-release`
Release base: `7221b55b5a5ec9648a8098c6c0fb3324fad7a48f`
Bucket final: `gs://labios-argentos-vsr-clean-v1/`

## Seguridad verificada

Comandos de lectura ejecutados antes de borrar:

```powershell
git branch --show-current
git status --short
git ls-remote --heads origin feature/full-clean-release
git rev-parse HEAD
gcloud.cmd compute instances list --format="table(name,zone,status,machineType)"
gcloud.cmd compute disks list --format="table(name,zone,status,sizeGb)"
gcloud.cmd compute addresses list --format="table(name,region,status,address)"
```

Resultado:

- Rama actual: `feature/full-clean-release`.
- HEAD local: `7221b55b5a5ec9648a8098c6c0fb3324fad7a48f`.
- `origin/feature/full-clean-release`: `7221b55b5a5ec9648a8098c6c0fb3324fad7a48f`.
- No se listaron instancias, discos ni direcciones estaticas en Compute.
- No se modifico ni se descargo nada del bucket.
- No se toco `.git/`, Git, gcloud, Python global, Node global ni ffmpeg global.

## Espacio liberado

Medicion antes:

- Working tree sin `.git`: 14.0 GB.
- `.git`: 10.0 GB.
- Total repo local incluyendo `.git`: 24.0 GB.
- Cache HuggingFace de usuario asociada a faster-whisper: 4.8 GB.

Medicion despues:

- Working tree sin `.git`: 425.6 MB.
- `.git`: 10.0 GB.
- Total repo local incluyendo `.git`: 10.4 GB.
- Cache HuggingFace de usuario: 5.5 KB.

Liberado:

- Repo local sin `.git`: ~13.6 GB.
- Modelos/caches faster-whisper globales claramente asociados al trabajo: ~4.8 GB.
- Total local aproximado liberado: ~18.4 GB.

## Paths borrados del repo local

| Path | Motivo | Reemplazo/canonico |
| --- | --- | --- |
| `data_release/local_sources` | Videos/audios fuente descargados localmente | Bucket `argentina/new_discovery/source_videos` y `source_audio` |
| `data_release/work` | Workdir temporal de reconstruccion/procesamiento | Manifests y reportes finales en `data_release/` y bucket |
| `data_release/cache` | Cache local | Regenerable |
| `data_release/logs` | Logs locales de corrida | Reportes finales versionados |
| `data_release/source_metadata/subtitles` | Subtitulos auxiliares | No requerido para consumir el release |
| `data_discovery/outputs/samples` | Samples de auditoria | Regenerable desde scripts de discovery |
| `data_discovery/outputs/contact_sheets` | Previews de auditoria | Regenerable |
| `evaluation/outputs/azzaro_whisper/videos` | Video local de analisis | Regenerable |
| `evaluation/outputs/batch_vsr/preprocessing_variant_smoke` | Outputs smoke/previews | Regenerable |
| `evaluation/outputs/batch_vsr/preprocessing_variant_preview` | Previews locales | Regenerable |
| `segmentacion_oraciones/outputs` | Corridas locales | Regenerable |
| `data_cleaning_clean_v1/outputs` | Outputs temporales de limpieza | Resultados aplicados en manifests finales |
| `data/processed/lip_rois` | ROIs locales pesados | Bucket `rois_npz` y final manifests |
| `data/clips` | Clips locales trackeados historicos | Bucket y manifests finales |
| `dataset` | Dataset local historico trackeado | Bucket y `final_train_manifest_clean_gpt_v1.csv` |
| `data/metadata` | Manifests/intermedios reemplazados | `data_release/*.csv` y reportes finales |
| `data_cleaning_clean_v1/video_jobs` | Prompts/jobs manuales ya aplicados | `clean_gpt_manifest.csv` y reportes de GPT |
| `data_cleaning_clean_v1/manual_gpt_handoff` | Handoff manual ya aplicado | Reportes y manifests finales |
| `data_cleaning_clean_v1/raw_outputs` | Raw outputs GPT ya aplicados/validados | `clean_gpt_manifest.csv` |
| `data_cleaning_clean_v1/validated` | Outputs validados ya aplicados | `clean_gpt_manifest.csv` |
| `data_cleaning_clean_v1/clean_gpt_v1` | Textos limpios locales ya integrados | Bucket/manifests finales |

## Caches y software

Borrado:

- `__pycache__/` en modulos del repo.
- `.jupyter_logs/`.
- `C:\Users\bianc\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3`.
- `C:\Users\bianc\.cache\huggingface\hub\models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`.
- `C:\Users\bianc\.cache\huggingface\hub\models--Systran--faster-whisper-small`.

Preservado:

- Python global y paquetes instalados globalmente.
- Git/gcloud/ffmpeg global.
- `.git/`.
- Cache pip global, porque no es especifica del proyecto.
- HuggingFace cache global no relacionada con faster-whisper.

No existian dentro del repo:

- `.venv/`, `venv/`, `env/`, `.conda/`.
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`.

## Preservado

Se preservaron codigo fuente, scripts, notebooks y docs finales. En particular:

- `README_DATASET.md`
- `OPEN_ITEMS_DATASET.md`
- `data_release/reports/HOW_TO_USE_BUCKET.md`
- `data_release/final_release_manifest.csv`
- `data_release/final_train_manifest_clean_gpt_v1.csv`
- `data_release/final_eval_manifest_clean_gpt_v1.csv`
- `data_release/clean_gpt_manifest.csv`
- `data_release/reports/`
- `data_release/scripts/`
- `data_cleaning_clean_v1/reports/`
- `data_cleaning_clean_v1/rejected_patches.jsonl`

## Validacion final

Comandos ejecutados despues del cleanup:

```powershell
python -m compileall data_cleaning_clean_v1 data_release data_discovery
python -m unittest discover data_discovery\tests
git diff --check
git diff --cached --check
```

Resultado:

- `compileall`: OK.
- `unittest discover data_discovery\tests`: 8 tests, OK.
- `git diff --check`: OK; solo avisos CRLF sobre archivos existentes.
- `git diff --cached --check`: OK.

## Como recuperar datos

Abrir primero:

```text
gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv
```

Para entrenar VSR, usar filas con:

- `usable_for_training=true`
- `npz_path` no vacio

Texto recomendado:

- `selected_training_text`

Guia de uso:

- `data_release/reports/HOW_TO_USE_BUCKET.md`
- `README_DATASET.md`

## Pendiente explicito

- `.git/` sigue pesando ~10.0 GB porque conserva historia y packs del repo. No se limpio por seguridad.
- No se borraron ramas remotas.
- `main` no se mergeo ni se force-pusheo.
