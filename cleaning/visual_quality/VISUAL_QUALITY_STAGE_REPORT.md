# Cierre de etapa visual / calidad de datos

## Objetivo

Esta etapa tuvo un objetivo acotado: auditar la calidad visual de los ROIs que consume el
modelo VSR y separar los clips claramente problematicos sin convertir el trabajo en un
proyecto nuevo de vision artificial.

El criterio fue conservador: la auditoria ayuda a diagnosticar y proponer un filtro
candidato, pero no reemplaza una evaluacion VSR real con WER/CER.

## Input real

La entrada real de entrenamiento/evaluacion son clips con texto alineado y ROIs labiales
96x96 en `.npz`. El split canonico vive en:

```text
vsr/splits/splits.csv
```

Cada fila apunta a:

- un ROI `.npz` en `data/processed/lip_rois/<source_id>/<clip>.npz`;
- un texto/transcripcion del clip;
- un split (`train`, `val`, `test`).

El VSR consume secuencias de frames, pero el label esta a nivel clip. Por eso esta etapa
evalua calidad del crop completo que ve el modelo, no decisiones frame-level.

## Que se hizo

Se implemento una auditoria visual ROI-aware. La decision principal fue separar senales
aplicables al crop ROI de senales que solo tienen sentido sobre video full-frame.

En modo ROI quedan fuera de decision senales como Haar, pose proxy, multi-face y speaker
mismatch proxy, porque no son confiables sobre crops de boca/cara inferior. La auditoria
se enfoca en brillo, blur, contraste, movimiento, visibilidad/actividad de boca, textura
y cortes visuales dentro del ROI.

Sobre el manifest sanity se agrego `policy analysis v2`, con:

- `review_severity`;
- `training_usability`;
- `policy_conservative`;
- `policy_moderate`;
- `policy_strict`.

Tambien se preparo una muestra estratificada de 300 clips para evaluacion barata:

- 100 `usable/keep`;
- 100 `questionable/keep`;
- 100 `bad_candidate/exclude`.

Finalmente se creo el puente visual -> VSR:

- exportador a scenario Gimeno `visual-quality-sample`;
- mapping `sample_id -> source_id + clip`;
- parser de `test.inf` a CSV con WER/CER por clip.

## Resultado principal

La politica candidata prudente es `policy_moderate_v2`:

| Grupo | Clips |
|---|---:|
| keep | 5732 |
| exclude | 218 |
| retencion | 96.34% |

`training_usability` queda distribuido asi:

| training_usability | Clips |
|---|---:|
| usable | 3330 |
| questionable | 2402 |
| bad_candidate | 218 |

Decision operativa:

- no usar `keep-only` del sanity como filtro final, porque descarta demasiado;
- conservar `usable + questionable`;
- tratar `bad_candidate` como candidato prudente a exclusion; la validacion con
  WER/CER que promoveria esta politica a filtro de produccion no se corrio (ver
  Validacion). El pipeline de entrenamiento sigue usando el manifest original
  (`keep`/`review`/`drop`).

## Validacion

Validacion local completada:

- auditoria ROI-aware y policy analysis v2 generados;
- notebooks 04/05 limpios y ejecutables;
- muestra VSR de 300 clips generada;
- scenario `visual-quality-sample` exportado localmente;
- mapping de 300 filas generado;
- puente `test.inf` -> WER/CER listo;
- tests unitarios verdes.

No se corrió el WER/CER final por grupo: requiere inferencia en VM con el checkpoint
v1 de Gimeno sobre el scenario exportado, y no se priorizó frente al resto del proyecto.
`policy_moderate_v2` queda como herramienta de diagnóstico documentada, no como filtro
adoptado.

## Cómo se reproduciría la comparación en VM

El flujo completo esta documentado en:

```text
vsr/RUNBOOK_visual_quality_eval.md
```

Comando de export local o en VM:

```bash
python -m cleaning.visual_quality.src.export_visual_quality_vsr_scenario \
  --sample data/metadata/visual_quality_vsr_eval_sample.csv \
  --output-base ~/data \
  --mapping-output data/metadata/visual_quality_vsr_eval_mapping.csv \
  --database Rioplatense \
  --scenario visual-quality-sample
```

Comando de inferencia en el repo de Gimeno:

```bash
CKPT=~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth

python vsr_main.py \
  --database Rioplatense \
  --scenario visual-quality-sample \
  --load-vsr "$CKPT" \
  --output-dir ./spanish-benchmark/rioplatense/visual_quality_sample/
```

Salida esperada:

```text
./spanish-benchmark/rioplatense/visual_quality_sample/inference/test.inf
./spanish-benchmark/rioplatense/visual_quality_sample/inference/test.wer
```

Parser local:

```bash
python -m cleaning.visual_quality.src.visual_quality_vsr_results \
  --inf data/metadata/visual_quality_sample.test.inf \
  --mapping data/metadata/visual_quality_vsr_eval_mapping.csv \
  --output data/metadata/visual_quality_vsr_eval_results.csv
```

El CSV resultante tiene:

```text
source_id,clip,split,policy_moderate,training_usability,reference,hypothesis,wer,cer
```

La comparacion relevante es WER/CER por `policy_moderate` y por
`training_usability`. No sacar conclusiones si algun grupo tiene menos de 30 clips.

## Decision de alcance

Queda fuera de esta etapa:

- resegmentar clips;
- active speaker detection;
- descarte frame-level;
- weighted training;
- curriculum learning;
- entrenamiento de variantes;
- nuevas heuristicas visuales.

La etapa visual queda cerrada. La comparación WER/CER por grupo que promovería
`policy_moderate_v2` a filtro de producción no se ejecutó (ver Validación); de
retomarse, el método sería: (1) correr inferencia VSR sobre `visual-quality-sample`,
(2) parsear `test.inf` con `visual_quality_vsr_results.py`, (3) comparar WER/CER por
`policy_moderate` y `training_usability`, y solo si `bad_candidate` muestra WER/CER
claramente peor, evaluar entrenar baseline vs filtered.
