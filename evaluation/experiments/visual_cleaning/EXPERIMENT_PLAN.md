# Experimentos: dataset original vs cleaning visual

## Hipotesis

Excluir solo los clips `bad_candidate` del entrenamiento puede mejorar WER/CER si esos
clips aportan ruido visual real. Como la politica moderada retiene la mayoria del
dataset, el efecto esperado puede ser chico; por eso la comparacion valida debe hacerse
en el mismo test original completo.

## Experimento A: baseline_original

- Train: `vsr_models/splits/train.csv` completo.
- Val: `vsr_models/splits/val.csv` completo.
- Test principal: `vsr_models/splits/test.csv` completo.
- Objetivo: linea base VSR entrenada con el dataset original.
- Metricas: WER, CER, WER/CER por `training_usability` si el resultado se puede
  matchear contra el manifest visual.

## Experimento B: visual_cleaned

- Train: `evaluation/outputs/visual_cleaning/manifests/visual_cleaned_train.csv`.
- Val: `evaluation/outputs/visual_cleaning/manifests/visual_cleaned_val.csv`.
- Test principal:
  `evaluation/outputs/visual_cleaning/manifests/visual_cleaned_test_original.csv`.
- Regla: el test principal no se filtra. Incluye el test original completo.
- Objetivo: medir si excluir `training_usability == bad_candidate` mejora el modelo.
- Metricas: WER, CER, WER/CER por `training_usability`, comparacion contra
  `baseline_original`.

Decision de val: para esta etapa se conserva val original completo. Es una comparacion
mas honesta porque evita optimizar solo sobre clips visualmente faciles. Si el loop de
training exigiera val filtrado, deberia generarse como auxiliar y documentarse separado,
sin reemplazar el test principal.

## Experimento C: llm_corrector_eval

- No entrena VSR.
- No modifica labels de entrenamiento.
- Toma predicciones de A/B, referencias reales y produce salidas corregidas.
- Compara raw vs corrected con WER/CER, tasa de edicion y casos empeorados.

## Comparaciones validas

- A vs B usando el mismo test original completo.
- A vs B con grupos `training_usability` solo si cada grupo tiene al menos 30 clips.
- Raw VSR vs LLM-corrected sobre las mismas predicciones y referencias.

## Fuera de alcance

- No se agregan nuevas politicas visuales.
- No se resegmentan clips.
- No se entrena en local si requiere VM/GPU.
- No se corre corrector LLM real hasta tener script/configuracion explicita.
- No se usa el test filtrado como numero titular.
