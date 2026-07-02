# Evaluacion VSR de muestra visual

Este runbook conecta la muestra visual `visual_quality_vsr_eval_sample.csv` con el
evaluador de Gimeno (`vsr_main.py`) para medir WER/CER por grupo visual sin entrenar.

## 1. Formato de scenario Gimeno

El loader de Gimeno construye las rutas asi:

```text
../data/<database>/ROIs/<spk>/<sampleID>.npz
../data/<database>/transcriptions/<spk>/<sampleID>.txt
../data/<database>/splits/<scenario>/test<database>.csv
```

Para este repo usamos:

```text
database = Rioplatense
scenario = visual-quality-sample
split VSR = test
```

Entonces el archivo que lee `--scenario visual-quality-sample` es:

```text
../data/Rioplatense/splits/visual-quality-sample/testRioplatense.csv
```

Ese CSV tiene una sola columna:

```csv
sampleID
```

El parche local `evaluation/gimeno_patches/aplicar_parches.py` registra `Rioplatense`
con `delimiter=5`. Por eso cada `sampleID` debe terminar con un sufijo de 5 caracteres,
por ejemplo `vq01_0000`; el speaker queda como `vq01`.

`vsr_main.py` escribe `inference/test.inf` con una linea `reference#hypothesis` por
clip, en el mismo orden que `testRioplatense.csv`. El cruce de vuelta a
`source_id + clip` se hace por orden usando el mapping generado abajo.

## 2. Exportar el scenario

Desde la raiz de este repo, con ROIs y textos disponibles:

```bash
python -m data_cleaning.src.export_visual_quality_vsr_scenario \
  --sample data/metadata/visual_quality_vsr_eval_sample.csv \
  --output-base evaluation/data/visual_quality_vsr_eval \
  --mapping-output data/metadata/visual_quality_vsr_eval_mapping.csv \
  --database Rioplatense \
  --scenario visual-quality-sample
```

Salidas locales:

```text
evaluation/data/visual_quality_vsr_eval/Rioplatense/ROIs/<spk>/<sampleID>.npz
evaluation/data/visual_quality_vsr_eval/Rioplatense/transcriptions/<spk>/<sampleID>.txt
evaluation/data/visual_quality_vsr_eval/Rioplatense/splits/visual-quality-sample/testRioplatense.csv
data/metadata/visual_quality_vsr_eval_mapping.csv
```

El mapping contiene:

```csv
sample_id,source_id,clip,split,scenario_split,path_roi,path_text,policy_moderate,training_usability,review_score,quality_score
```

## 3. Llevarlo a la VM

Opcion recomendada si el repo y los ROIs ya estan en la VM: correr el exportador alla y
escribir directo en `~/data`, porque el repo de Gimeno busca `../data` relativo a su
raiz:

```bash
python -m data_cleaning.src.export_visual_quality_vsr_scenario \
  --sample data/metadata/visual_quality_vsr_eval_sample.csv \
  --output-base ~/data \
  --mapping-output data/metadata/visual_quality_vsr_eval_mapping.csv \
  --database Rioplatense \
  --scenario visual-quality-sample
```

Opcion alternativa: exportar localmente y copiar la carpeta:

```bash
rsync -av evaluation/data/visual_quality_vsr_eval/Rioplatense/ \
  <vm>:~/data/Rioplatense/
```

No versionar `evaluation/data/visual_quality_vsr_eval/`: contiene copias de ROIs `.npz`
y se regenera con el comando de export.

## 4. Correr inferencia

Desde la raiz del repo de Gimeno en la VM, con el entorno `vsr-factors` activo y los
parches aplicados:

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

`test.inf` tiene el formato:

```text
reference#hypothesis
```

## 5. Convertir a WER/CER por clip

Copiar `test.inf` de vuelta al repo si se corrio en VM:

```bash
scp <vm>:~/evaluating-end2end-spanish-lipreading/spanish-benchmark/rioplatense/visual_quality_sample/inference/test.inf \
  data/metadata/visual_quality_sample.test.inf
```

Despues ejecutar:

```bash
python -m data_cleaning.src.visual_quality_vsr_results \
  --inf data/metadata/visual_quality_sample.test.inf \
  --mapping data/metadata/visual_quality_vsr_eval_mapping.csv \
  --output data/metadata/visual_quality_vsr_eval_results.csv
```

Salida:

```csv
source_id,clip,split,policy_moderate,training_usability,reference,hypothesis,wer,cer
```

`wer` y `cer` son fracciones por clip (`0.0` perfecto, `1.0` error equivalente a toda
la referencia).

## 6. Interpretacion

Comparar promedios y distribuciones por:

- `policy_moderate`: `keep` vs `exclude`;
- `training_usability`: `usable`, `questionable`, `bad_candidate`;
- `split`: para detectar si el efecto depende de train/val/test original.

No sacar conclusiones de un grupo con menos de 30 ejemplos. El parser imprime
`warnings` si `policy_moderate` o `training_usability` quedan por debajo de ese piso.

La politica visual queda mejor justificada si `bad_candidate/exclude` tiene WER/CER
claramente peor que `usable/keep` y `questionable/keep`. Si no hay diferencia clara, no
usar `policy_moderate_v2` como filtro final de entrenamiento todavia.
