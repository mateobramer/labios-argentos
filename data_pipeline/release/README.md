# Data release clean bucket v1

> **Nota de contexto (reorganización 2026-07):** este directorio fue portado desde
> `feature/full-clean-release` (PR #25). Los manifests grandes no forman parte del árbol
> actual para mantener liviano el working tree. Se recuperan con
> `git show dataset-clean-v1:data_release/<nombre>.csv > <nombre>.csv`
> o desde el bucket clean-v1. La política vigente deja código, reportes y manifests chicos
> en Git, y datos/manifests pesados en bucket. Ver
> [`docs/DATA_AND_ARTIFACTS.md`](../../docs/DATA_AND_ARTIFACTS.md).

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
