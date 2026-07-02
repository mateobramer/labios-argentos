# data_cleaning

Auditoria y limpieza del dataset crudo.

Este modulo sirve para revisar si los pares `clip_NNNN.mp4` / `clip_NNNN.txt`
son buenos datos antes de pasarlos al preprocesamiento visual o al entrenamiento.

## Que va aca

- inventarios de clips, textos y fuentes;
- chequeos de pares faltantes entre `.mp4` y `.txt`;
- duracion de clips y textos sospechosos;
- revision de cortes raros, por ejemplo clips que empiezan o terminan en mitad de una palabra;
- manifests de calidad con estados como `keep`, `review` y `drop`;
- scripts o notebooks de limpieza cuando tengan codigo real y resultados reproducibles.

## Que no va aca

- deteccion de rostro, landmarks o recorte de labios: eso va en `visual_preprocessing/`;
- entrenamiento de modelos VSR: eso va en `vsr_models/` si se agrega al repo;
- logica de LLM/correccion en tiempo real: eso va en `realtime/` si se agrega al repo.

## Detector de clips malos (`src/detectar_clips_malos.py`)

Auditoria automatica de los ROIs labiales **96x96** que produce el preprocesamiento
visual. Corre sobre `data/processed/lip_rois/` (lo que efectivamente ve el modelo) y
**no usa MediaPipe**: trabaja directo sobre los pixeles del recorte, asi que es barato
(solo `cv2` + `numpy`). Detecta los tres problemas que mas dañan el fine-tuning:

| Problema | Metrica | Estado |
|---|---|---|
| **Negro / oscuro** (placa, fundido, camara tapada) | `luma_media`, `frac_oscuros` | `drop` |
| **Congelado** (imagen estatica, sin habla real) | `movimiento_global` (diff temporal media) | `drop` |
| **Boca inactiva / tapada** (mano, microfono, mala alineacion, silencio) | `actividad_boca` (std temporal de la region central) + `textura_boca` (varianza de Laplaciano) | `review` |

La oclusion de boca es un proxy **heuristico**: MediaPipe alucina landmarks aunque la
boca este tapada, asi que el preproc no la descarta; aca la cazamos por baja
actividad+textura en la region de la boca y la marcamos `review` (no `drop`), para
mirarla a ojo antes de decidir.

Los umbrales (en `src/detectar_clips_malos.py`) estan **calibrados sobre las 9 fuentes
ya procesadas**: los de `drop` (negro/congelado) son red de seguridad para futuras
fuentes; los de `review` se situan en el piso real de la distribucion para surfacear la
cola dudosa. En este dataset: **1683 `keep`, 21 `review`, 0 `drop`** (ya estaba limpio
de negro/congelado por el filtro de 80% de cara del preproc).

### Uso

```bash
# 1. Auditar todos los ROIs -> escribe el manifest de auditoria
python -m data_cleaning.src.detectar_clips_malos

# Una sola fuente
python -m data_cleaning.src.detectar_clips_malos "<titulo>"

# 2. Materializar el dataset final curado (copia los `keep` -> dataset/)
python -m data_cleaning.src.detectar_clips_malos --materializar
```

Salida:

```text
data/metadata/auditoria_clips_manifest.csv   # un unico manifest: keep/review/drop + metricas + razones
dataset/<titulo>/clip_NNNN.mp4 | .txt         # set final curado (solo `keep`)
dataset/manifest.csv                          # inventario del set curado
```

El paso de materializar **no borra** nada de `data/processed/lip_rois/` (ese es el
insumo crudo y se conserva). Los `review` quedan retenidos: no se copian a `dataset/`
hasta decidirlos a mano.

## Auditoria visual offline fuerte (`src/visual_quality_audit.py`)

Primera etapa de limpieza visual offline para decidir `keep | review | drop` antes de
entrenar VSR. En modo ROI (`input_kind=roi_npz`) mide principalmente la calidad del
crop que ve el modelo: legibilidad, brillo, blur, contraste, movimiento, visibilidad y
actividad de boca, y cortes visuales dentro del ROI.

La auditoria ROI **no puede medir de forma confiable** multiples caras, pose/frontalidad,
tracking de cara completa ni active speaker detection. Esas senales requieren una
auditoria separada sobre full-frame/raw video. En ROI mode quedan trazadas como
`not_applicable` o `experimental_metric`, pero no participan en la decision automatica.

No modifica datos originales. Lee los ROIs `.npz` de `data/processed/lip_rois/` cuando
existen, porque eso es lo que ve el VSR. Si el checkout local no los tiene, cae a
`data/clips/` y deja trazado `input_kind=raw_clip`, `audit_confidence=low` y
`audit_scope=fallback_clip_crudo_no_equivale_al_roi_del_vsr`.

Uso:

```bash
python -m data_cleaning.src.visual_quality_audit \
  --split vsr_models/splits/splits.csv \
  --output data/metadata/visual_quality_manifest_smoke.csv \
  --keep-output data/metadata/visual_quality_keep_manifest_smoke.csv \
  --keep-review-output data/metadata/visual_quality_keep_review_manifest_smoke.csv \
  --sample-strategy stratified \
  --clips-per-source 5 \
  --max-clips 100 \
  --allow-raw-fallback \
  --seed 42
```

Para correr un split completo, omitir `--sample-strategy`, `--clips-per-source` y
`--max-clips`.

Columnas principales:

```text
split, source_id/titulo, clip, path_roi, path_video, path_text,
input_kind, audit_confidence, audit_scope, n_frames,
frame_read_ratio, brightness_score, blur_score, motion_score,
mouth_visibility_score, mouth_activity_score, mouth_texture_score,
scene_cut_score, scene_cut_count, face_count_score, track_stability_score,
multi_face_risk, face_boxes_summary, pose_score, pose_available, pose_reason,
speaker_mismatch_risk, speaker_mismatch_available, active_speaker_available,
quality_score, quality_bucket, decision, decision_confidence,
metric_scope, used_for_decision_reasons,
hard_fail_reasons, review_reasons, experimental_reasons,
non_applicable_reasons, invalid_for_input_reasons, unavailable_signals
```

Interpretacion:

- `drop`: solo hard fail claro del ROI (`input_visual_missing`, `video_no_legible`,
  `sin_frames`, `oscuridad_extrema`, `freeze_extremo_confirmado` en ROI, boca totalmente
  no visible en ROI confiable).
- `review`: senales dudosas o accionables manualmente del ROI (`blur`, `blur_extremo`
  sin otro hard fail, `movimiento_bajo`, `corte_escena`, `boca_inactiva`,
  `boca_tapada_o_poco_visible`, `baja_textura_boca`, `contraste_bajo`, etc.).
- `keep`: sin alertas visuales fuertes.

En `roi_npz`, Haar, multi-face, pose, tracking de cara y speaker mismatch proxy quedan
fuera de decision (`non_applicable_reasons` / `invalid_for_input_reasons`). Active
speaker detection queda explicitamente no disponible: `active_speaker_available=false`.

Previews opcionales:

```bash
python -m data_cleaning.src.visual_quality_audit \
  --split vsr_models/splits/splits.csv \
  --output data/metadata/visual_quality_manifest.csv \
  --sample-strategy stratified \
  --clips-per-source 2 \
  --max-clips 30 \
  --preview-dir data_cleaning/outputs/visual_quality_audit/previews
```

No commitear previews masivas. Usarlas como apoyo visual local.

### Preflight de ROIs

Antes de interpretar resultados para VSR, correr:

```bash
python -m data_cleaning.src.visual_quality_audit \
  --split vsr_models/splits/splits.csv \
  --preflight-only
```

El preflight informa:

- si existe `data/processed/lip_rois/`;
- cuantos `.npz` hay en disco;
- cobertura ROI sobre el split/muestra;
- cuantos clips caerian a `raw_clip`;
- `run_mode`: `smoke_raw_fallback`, `roi_audit` o `mixed_audit`;
- si la corrida es interpretable para VSR.

Si la cobertura ROI es baja, los resultados son solo smoke test: validan codigo y
previews, pero no sirven para decidir filtros de entrenamiento.

### Como correr auditoria real con ROIs

Primero traer los ROIs del bucket (fuera del script; la auditoria no descarga datos):

```bash
gcloud auth login
gcloud config set project labios-argentos-vsr
gcloud storage rsync -r gs://labios-argentos-vsr-data/lip_rois ./data/processed/lip_rois
```

Despues exigir cobertura de ROI:

```bash
python -m data_cleaning.src.visual_quality_audit \
  --split vsr_models/splits/splits.csv \
  --require-roi \
  --min-roi-coverage 0.8 \
  --output data/metadata/visual_quality_manifest_full_roi_sanity.csv \
  --keep-output data/metadata/visual_quality_keep_manifest_full_roi_sanity.csv \
  --keep-review-output data/metadata/visual_quality_keep_review_manifest_full_roi_sanity.csv
```

No pisar `visual_quality_manifest_full_roi.csv`: sirve como corrida historica para
comparar contra la calibracion sanity.

### Analisis de politicas candidatas

La auditoria sanity ROI corrige el scope conceptual, pero **no es automaticamente un
filtro final**. En la corrida actual no hay `drop`: `keep-only` descarta demasiado
(`keep_pct` cercano a 56%) y `keep+review` conserva todo. Por eso el paso siguiente es
generar politicas candidatas desde el manifest sanity, sin tocar splits ni copiar datos:

```bash
python -m data_cleaning.src.visual_quality_policy_analysis \
  --input data/metadata/visual_quality_manifest_full_roi_sanity.csv \
  --output data/metadata/visual_quality_policy_analysis_v2.csv \
  --moderate-keep-output data/metadata/visual_quality_policy_moderate_keep_v2.csv \
  --strict-keep-output data/metadata/visual_quality_policy_strict_keep_v2.csv
```

Salidas:

```text
data/metadata/visual_quality_policy_analysis_v2.csv        # manifest sanity + severity + utilidad + politicas
data/metadata/visual_quality_policy_moderate_keep_v2.csv  # clips keep bajo policy_moderate_v2
data/metadata/visual_quality_policy_strict_keep_v2.csv    # clips keep bajo policy_strict_v2
```

Columnas agregadas:

- `review_severity`: `none | low | medium | high`;
- `review_score` y `review_reason_group`;
- `training_usability`: `usable | questionable | bad_candidate`;
- `training_usability_reasons`: razones orientadas a utilidad VSR (`mouth_visible`,
  `low_mouth_motion`, `blur_low_texture`, `scene_discontinuity`, `quality_tail`, etc.);
- `policy_conservative`: excluye solo hard fails reales;
- `policy_moderate`: excluye solo `bad_candidate` por utilidad VSR. Blur, baja textura
  o `quality_tail` aislados suben severidad, pero no excluyen si la boca sigue visible y
  temporalmente consistente;
- `policy_strict`: variante agresiva solo para analisis de sensibilidad.

`keep-only` actual no se recomienda como filtro de entrenamiento. `policy_moderate_v2`
es un candidato para evaluar visualmente y con WER/CER, no un filtro final. `policy_strict`
no debe usarse como recomendacion automatica.

Antes de gastar una VM entrenando, la evaluacion barata recomendada es correr inferencia
con el modelo v1 y cruzar WER/CER por clip. El helper acepta un CSV futuro con:

```csv
source_id,clip,split,reference,hypothesis,wer,cer
```

o minimo:

```csv
source_id,clip,wer,cer
```

El experimento VM de entrenamiento recien se justifica si una politica candidata muestra
correlacion con WER/CER o si la inspeccion visual confirma que los excluidos son
claramente malos. En particular, hace falta inferencia del modelo v1 sobre una muestra
estratificada que incluya `policy_moderate_v2=keep` y `policy_moderate_v2=exclude`.

### Muestra estratificada para evaluacion VSR

Para preparar una evaluacion barata sin entrenar ni rearmar clips:

```bash
python -m data_cleaning.src.visual_quality_eval_sample \
  --policy-analysis data/metadata/visual_quality_policy_analysis_v2.csv \
  --output data/metadata/visual_quality_vsr_eval_sample.csv \
  --per-group 100 \
  --seed 42
```

La muestra toma cada combinacion existente de `policy_moderate_v2` y
`training_usability` como grupo principal, e intenta balancear dentro de cada grupo por
`split` y `source_id`. Si un grupo tiene menos de `--per-group`, incluye todos sus clips.

Columnas:

```text
source_id,clip,split,path_roi,path_text,policy_moderate_v2,training_usability,
review_severity,review_score,exclusion_reasons,quality_score,
mouth_activity_score,mouth_visibility_score,scene_cut_score,blur_score
```

La salida esperada de una inferencia futura para cruzar WER/CER por clip es:

```csv
source_id,clip,split,reference,hypothesis,wer,cer
```

No hay todavia un CLI directo que tome `visual_quality_vsr_eval_sample.csv` y ejecute el
modelo v1. El flujo existente de Gimeno evalua un `scenario` exportado al layout de
`evaluation/src/exportar_para_gimeno.py` y luego corre, en la VM/env `vsr-factors`:

```bash
CKPT=~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth
python vsr_main.py --database Rioplatense --scenario <scenario_muestra> \
  --load-vsr $CKPT \
  --output-dir ./spanish-benchmark/rioplatense/visual_quality_sample/
```

Ese comando produce `inference/test.inf` (`ref#hyp`) y `test.wer`; para este diagnostico
hay que convertir `test.inf + mapeo.csv` al CSV por clip anterior y comparar WER/CER por
`policy_moderate_v2` y `training_usability`. No se recomienda entrenar `actual` vs
`policy_moderate_v2` hasta tener esa comparacion por grupo.

### Smoke test local sin ROIs

Para validar infraestructura local sin sacar conclusiones de dataset:

```bash
python -m data_cleaning.src.visual_quality_audit \
  --split vsr_models/splits/splits.csv \
  --sample-strategy stratified \
  --clips-per-source 2 \
  --max-clips 60 \
  --allow-raw-fallback \
  --output data/metadata/visual_quality_manifest_smoke.csv
```

No mezclar conclusiones: `visual_quality_manifest_smoke.csv` valida el flujo; una corrida
con `*_full_roi.csv` valida el dataset visual que consume el VSR.

Notebooks de auditoria visual:

```text
data_cleaning/notebooks/04_auditoria_visual_offline.ipynb          # resumen/presentacion
data_cleaning/notebooks/05_diagnostico_politicas_visuales.ipynb   # diagnostico/debug
```

El notebook 04 es corto: responde que se audito, que politicas hay, cuanto retienen,
que ejemplos visuales representan cada caso y que decision accionable sale. El notebook
05 es largo: diagnostica por que se excluye cada clip, fuentes/splits afectados,
posibles falsos positivos, WER/CER disponible, muestra estratificada VSR y ejemplos por
razon. La logica larga de carga, tablas y contact sheets vive en
`data_cleaning/src/visual_quality_notebook_utils.py`; los notebooks quedan como capa de
presentacion/diagnostico.

El comando para pasar de smoke a auditoria real es:

```bash
gcloud storage rsync -r gs://labios-argentos-vsr-data/lip_rois ./data/processed/lip_rois
```

Luego correr la auditoria con `--require-roi --min-roi-coverage 0.8` y el analisis de
politicas v2. El notebook 04 carga el manifest sanity y
`visual_quality_policy_analysis_v2.csv`; el 05 profundiza el diagnostico.

Plan de comparacion en VM:

```text
primero: inferencia modelo v1 por clip sobre muestra estratificada keep/exclude de policy_moderate_v2
despues: entrenar actual vs policy_moderate_v2 solo si WER/CER o inspeccion visual justifican el filtro
```

## Notebook de revision visual

Para mirar a ojo los casos `review` (o sospechosos) con video + texto + razon, la
revision humana arranca en:

```text
data_cleaning/notebooks/01_revision_visual_mediapipe.ipynb
```

El notebook usa MediaPipe para detectar casos sospechosos para VSR:

- 0 caras aunque haya texto;
- mas de una cara;
- boca poco visible;
- boca muy descentrada.

No borra nada. Primero muestra video + texto + razon de revision. Evitar crear CSVs
paralelos por cada experimento; si hace falta guardar una decision, se agrega a un unico
manifest acordado en `data/metadata/`.

## Comparacion de modelos Whisper

El notebook:

```text
data_cleaning/notebooks/02_analisis_whisper_azzaro.ipynb
```

compara `turbo`, `small` y `large` sobre el mismo video, forma clips con timestamps
por palabra y pausas reales, y permite seleccionar manualmente la mejor transcripcion.
Los artefactos finales chicos se versionan en:

```text
data_cleaning/outputs/azzaro_whisper/final/
```

El video original, los modelos descargados y los clips descartados permanecen locales.

## Correcciones textuales asistidas por LLM

Para auditar texto clip por clip sin pisar los `.txt` originales, usar:

```text
data_cleaning/src/llm_text_corrections.py
data_cleaning/notebooks/03_revision_correcciones_llm.ipynb
```

Exportar un prompt para un LLM grande:

```bash
python -m data_cleaning.src.llm_text_corrections \
  --export-prompt \
  --split vsr_models/splits/val.csv \
  --source-id "CHARLA SOBRE EL AMOR Y EL DESAMOR" \
  --prompt-output data_cleaning/outputs/llm_text_corrections/charla_amor_desamor/gpt_clip_by_clip_prompt.md
```

Validar la respuesta JSON del LLM:

```bash
python -m data_cleaning.src.llm_text_corrections \
  --split vsr_models/splits/val.csv \
  --suggestions data_cleaning/outputs/llm_text_corrections/charla_amor_desamor/llm_suggestions.raw.json \
  --output-dir data_cleaning/outputs/llm_text_corrections/charla_amor_desamor
```

Salida:

```text
llm_suggestions.raw.json   # respuesta cruda del LLM
review_manifest.csv        # manifest auditable raw vs sugerido
review_manifest.jsonl      # mismo manifest en JSONL
summary.json               # resumen de acciones, riesgos y validaciones
```

Nada de esto modifica los textos originales. Las sugerencias quedan como propuestas
con decision automatica:

- `accept_correction`: cambio chico y confiable;
- `keep_raw`: no se cambia el texto original;
- `reject_suggestion`: se descarta la sugerencia y se conserva el raw.

El notebook de revision exporta clips de apoyo en:

```text
data_cleaning/outputs/llm_text_corrections/<fuente>/review_webm/
```

Esto evita el problema de audio AAC en el renderer de notebooks de VS Code. Para cada
caso genera:

- WebM de respaldo;
- MP4 H.264 con audio MP3 para reproducir dentro de VS Code;
- MP3 separado como fallback de audio;
- thumbnail y links externos al MP4 compatible, MP4 original, MP3 y WebM.
