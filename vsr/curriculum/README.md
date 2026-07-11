# vsr/curriculum/ — procesamiento ViSpeR-es para currículum de pre-entrenamiento

**Estado: pausado** (gate go/kill de [`docs/PLAN_CURRICULUM.md`](../../docs/PLAN_CURRICULUM.md);
el costo/beneficio quedó desfavorable cuando ViSpeR zero-shot superó al teacher ft05 —
ver [`docs/PROJECT_EVOLUTION.md`](../../docs/PROJECT_EVOLUTION.md)).

Scripts de la fase 1 del plan: convertir los datos oficiales de ViSpeR-es
(JSON + video-ids de YouTube) a clips/ROIs con el mismo warp mean-face de ft05.

| Script | Qué hace |
|---|---|
| `procesar_visper.py` | orquestador: descarga con yt-dlp, corta segmentos, croppea con landmarks de ViSpeR |
| `visper_a_clips.py` | JSON ViSpeR → clips alineados |
| `visper_crop_landmarks.py` | crop 96×96 usando los landmarks provistos por ViSpeR |

Los logs de procesamiento (`vsr/curriculum/logs/`) y `data/_visper_tmp/` no se versionan.
