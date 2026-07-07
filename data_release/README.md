# Data release clean bucket v1

Artefactos livianos para construir el bucket limpio `labios-argentos-vsr-clean-v1`.

Este directorio no contiene videos, audios, ROIs ni credenciales. Los manifests se
generan desde listados de GCS y archivos CSV chicos del bucket fuente.

Comando principal:

```bash
python data_release/scripts/build_release_manifests.py
```

Salidas esperadas:

- `data_release/argentina_existing_manifest.csv`
- `data_release/spanish_general_manifest.csv`
- `data_release/argentina_new_manifest.csv`
- `data_release/reports/manifest_build_report.md`
- `data_release/reports/failures.csv`
- `data_release/reports/bucket_build_report.md`
- `data_release/reports/cost_runtime_report.md`

Los listados crudos cacheados en `data_release/cache/` son regenerables y no se
versionan.
