# Data discovery

Modulo para buscar nuevas fuentes de datos VSR/lip-reading en espanol
rioplatense/argentino sin modificar el dataset actual.

La idea no es juntar links prometedores a ojo. El flujo deja evidencia reproducible:

1. buscar candidatos de YouTube sin descargar videos completos;
2. bajar samples cortos y distribuidos;
3. auditar visual/audio/contexto con metricas automaticas;
4. scorear y rankear;
5. estimar minutos utiles y clips aceptados;
6. aplicar caps de diversidad por fuente;
7. generar shortlist, rechazados, progreso al target e ingest plan.

## Estructura

```text
data_pipeline/discovery/
  README.md
  sources_seed.csv
  src/
    search_youtube_candidates.py
    audit_youtube_candidate.py
    score_candidates.py
    make_contact_sheets.py
    estimate_usable_clips.py
  outputs/
    candidate_videos.csv
    candidate_scores.csv
    shortlist_recommended.csv
    rejected_candidates.csv
    target_progress.md
    source_diversity_report.csv
    review_index.md
    ingest_plan_v1.md
    contact_sheets/
    sample_metadata/
```

`outputs/samples/` y contact sheets de imagen quedan ignorados por git. Los JSON/CSV/MD
livianos si pueden versionarse.

## Comandos

Buscar candidatos:

```bash
python -m data_pipeline.discovery.src.search_youtube_candidates --max-results 8
```

Auditar samples cortos:

```bash
python -m data_pipeline.discovery.src.audit_youtube_candidate --limit 20 --sample-count 3 --sample-seconds 24
```

ASR opcional sobre samples (descarga/carga el modelo elegido, CPU/int8):

```bash
python -m data_pipeline.discovery.src.audit_youtube_candidate --limit 20 --sample-count 3 --sample-seconds 24 --run-asr --asr-model small
```

La corrida base no fuerza ASR para evitar descargar modelos en loops locales. Si
`faster-whisper` no esta instalado, los JSON marcan `blocked_missing_asr_dependency`.

Scorear y crear reportes:

```bash
python -m data_pipeline.discovery.src.score_candidates --clips-per-minute 12
```

Generar contact sheets locales:

```bash
python -m data_pipeline.discovery.src.make_contact_sheets --accepted-only
```

Validar:

```bash
python -m compileall data_pipeline/discovery
python -m unittest discover data_pipeline/discovery/tests
```

## Scores

`total_score` usa:

```text
0.50 * visual_quality_score
+ 0.30 * audio_quality_score
+ 0.20 * context_score
```

Criterio minimo para `accept`:

- `total_score >= 85`;
- `visual_quality_score >= 80`;
- `audio_quality_score >= 75`;
- fuente argentina/rioplatense probable;
- sample audit OK.

No se bajan thresholds para llegar al numero.

## Estimacion de clips

`usable_minutes_estimate` no cuenta la duracion total del video. Usa:

```text
duration_minutes
* speech_presence_ratio
* mouth_visible_ratio
* single_speaker_ratio
* visual_accept_ratio
```

`accepted_clips_estimate` multiplica minutos utiles por `clips_per_minute_estimate`.
La configuracion inicial usa `12 clips/min`, conservadora para clips de 3-10 s.

## Diversidad

Caps por fuente:

- `max_accepted_clips_per_source = 1800`;
- `max_usable_minutes_per_source = 60`;
- objetivo minimo: 10 fuentes;
- ideal: 15+ fuentes.

Si una fuente supera el cap, los videos siguientes quedan como backup/manual review.

## Bucket / datos existentes

Estado revisado durante la auditoría de 2026-07 con comandos read-only:

```bash
gcloud storage ls gs://labios-argentos-vsr-data/
gsutil du -s gs://labios-argentos-vsr-data/lip_rois gs://labios-argentos-vsr-data/models
```

Hallazgos:

- Raiz del bucket: `adaptacion/`, `corrector/`, `lip_rois/`, `lip_rois_full/`, `models/`.
- `lip_rois/`: ~4.4 GB.
- `models/`: ~2.1 GB.
- `adaptacion/`: checkpoints, logs, scripts y ROIs de adaptacion.
- `corrector/`: pares train/val/test y modelo del corrector.
- `lip_rois_full/`: lista carpetas de ROIs por fuente; el listado corto mostro fuentes ya
  procesadas, pero `gcloud` corto por Unicode de consola Windows al imprimir un emoji.
- No se encontro en la raiz del bucket un set claro de raw videos/clips/audio nuevos para
  discovery.

Conclusion:

- ROIs solos no alcanzan para transcript discovery/cleaning porque no incluyen audio,
  contexto ni metadata original completa.
- Para nuevos datos hacen falta videos/clips/audio + metadata + transcripts.
- Si el bucket tiene clips/audio en una subcarpeta no listada, se puede incorporar en una
  ronda posterior; con la evidencia actual, discovery nuevo debe reconstruirse desde
  URLs/videos originales.
- No subir raw videos nuevos al bucket sin aprobacion explicita.

Comando de verificacion sugerido cuando haya credenciales GCP:

```bash
gcloud storage ls -r gs://labios-argentos-vsr-data/**
```
