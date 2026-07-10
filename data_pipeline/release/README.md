# Data release clean bucket v1

> **Nota de contexto (limpieza 2026-07, `chore/repo-cleanup-safe-v2`):** este directorio
> fue portado desde la rama `feature/full-clean-release` (PR #25). Los **manifests
> grandes (≥1 MB, 12 CSVs, ~127 MB)** — `final_release_manifest.csv`,
> `final_train_manifest_clean_gpt_v1.csv`, `clean_gpt_manifest.csv`,
> `alignment_manifest.csv`, `asr_large_turbo_manifest.csv`,
> `argentina_existing_manifest.csv`, `existing_reconstruction_manifest.csv`,
> `new_discovery_{asr,clip,roi}_manifest.csv`, `spanish_general{,_asr}_manifest.csv` —
> **no están en esta rama** para no inflar el working tree. Recuperarlos:
> `git show dataset-clean-v1:data_release/<nombre>.csv > <nombre>.csv`
> (el tag `dataset-clean-v1` los preserva) o regenerarlos con los scripts de acá.
> Ver [`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md).
> Ojo: esa rama también propone una política de datos (todo al bucket) que main no
> adoptó — es decisión abierta, ver [`docs/NEXT_STEPS.md`](../docs/NEXT_STEPS.md).

Artefactos livianos para construir el bucket limpio `labios-argentos-vsr-clean-v1`.

Este directorio no contiene videos, audios, ROIs ni credenciales. Los manifests se
generan desde listados de GCS y archivos CSV chicos del bucket fuente.

Comando principal:

```bash
python data_pipeline/release/scripts/build_release_manifests.py
```

Salidas esperadas:

- `data_pipeline/release/argentina_existing_manifest.csv`
- `data_pipeline/release/spanish_general_manifest.csv`
- `data_pipeline/release/argentina_new_manifest.csv`
- `data_pipeline/release/reports/manifest_build_report.md`
- `data_pipeline/release/reports/failures.csv`
- `data_pipeline/release/reports/bucket_build_report.md`
- `data_pipeline/release/reports/cost_runtime_report.md`

Los listados crudos cacheados en `data_pipeline/release/cache/` son regenerables y no se
versionan.
