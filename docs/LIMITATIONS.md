# Limitations — qué NO hace el sistema (honesto)

Este proyecto es un prototipo de investigación e ingeniería. Sus límites, sin marketing:

## Del modelo y la tarea

- **Visual-only.** Lee labios sin audio. La lectura de labios pura es un problema
  intrínsecamente difícil y ambiguo (muchos fonemas comparten visema).
- **Offline / bidireccional.** El modelo base (ViSpeR) mira el segmento completo. **No es
  streaming causal** cuadro a cuadro: la demo aproxima tiempo real cortando el habla en
  segmentos por pausas de labios e infiriendo cada segmento entero.
- **Inferencia por segmentos, no continua.** La latencia es ~fin-de-frase + ~1.1 s.

## De la precisión

- **WER ~26–30** en condiciones ideales (buena luz, boca frontal, habla clara);
  **~45** en YouTube variado. Ver [`RESULTS.md`](RESULTS.md).
- **El fine-tuning propio no supera al zero-shot de ViSpeR**: el techo lo pone la escala
  de pre-entrenamiento (794 h), no la arquitectura ni nuestras ~19 h de datos.

## Del corrector LLM

- **La corrección 1-best empeoró en todas las condiciones evaluadas.** El **n-best
  rescoring** produjo una mejora significativa en el régimen de CER bajo evaluado
  (~11, self-test): −3.04 WER. En los ensayos de CER más alto el beam no ofreció
  candidatos suficientes para recuperar el error. Detalle: [`RESEARCH.md`](RESEARCH.md).

## De la personalización

- Validada en profundidad con **un solo hablante** (n=1): mejora de 4.7 puntos de WER
  personal sin degradación del test general. La mejora personal todavía no fue
  estadísticamente significativa con la muestra actual y la generalización a más
  hablantes es trabajo pendiente.
- La captura de correcciones humanas existe como mínimo local (JSONL); todavía no
  alimenta un re-entrenamiento automático.

## De la plataforma y los datos

- El encoder acelerado requiere **Apple Silicon (MPS)**; en CPU el total sube a ~1.5 s/seg.
- No identificamos un corpus audiovisual público específicamente orientado al español rioplatense; el dataset es propio
  y su escala es la principal limitación de los modelos entrenados.
- El repositorio versiona solo artefactos livianos y una muestra mínima; el dataset
  completo, los ROIs y los pesos viven fuera de Git ([`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md)).

Próximos pasos para cada límite: [`FUTURE_WORK.md`](FUTURE_WORK.md).
