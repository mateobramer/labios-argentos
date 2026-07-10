# data_pipeline/ — recolección, corte y release del dataset

Todo el ciclo de datos ANTES de la limpieza: elegir fuentes, descargar, cortar,
inventariar y construir releases. La limpieza/calidad vive en [`cleaning/`](../cleaning/);
dónde vive cada dato (git vs bucket) en [`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md).

| Path | Etapa | Qué hace |
|---|---|---|
| `sources/` | 0 | CSVs de fuentes curadas a mano (gate 0) |
| `descargar_procesar.py` | 1 | YouTube → Whisper → ffmpeg → clips alineados mp4+txt |
| `discovery/` | 0b | búsqueda y scoring de fuentes nuevas (src+tests+outputs) |
| `release/` | R | manifests, reportes y scripts del release limpio v1 (tag `dataset-clean-v1`) |
| `inventory/` | R | inventario del bucket |

```bash
python data_pipeline/descargar_procesar.py <url> [...]      # etapa 1
python -m data_pipeline.discovery.src.score_candidates --help
```

**Regla dura de scraping**: sin cookies del browser ni rotación de IPs; fuentes de a
una con gates de calidad ([CONTRIBUTING](../CONTRIBUTING.md), [exp. 07](../docs/experiments/07_datos_y_scraping.md)).
