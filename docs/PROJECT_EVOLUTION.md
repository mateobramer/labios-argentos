# Project Evolution — evolución y alcance del proyecto

Reconocimiento visual del habla (VSR) en español rioplatense. Este doc explica **cómo
evolucionó el proyecto**: del pipeline original al alcance actual, con las decisiones que
cambiaron y su evidencia.

## Pipeline original (2026, primera mitad)

El plan inicial era un pipeline único estilo Auto-AVSR: construir un corpus audiovisual
rioplatense desde YouTube, preprocesarlo (ROI de boca 96×96), y fine-tunear un modelo
propio (50M de Gimeno, base LIP-RTVE) hasta tener el mejor VSR posible en el acento.
Los docs de esa etapa (FLUJO, ESTRUCTURA_PROYECTO, PIPELINE_PROYECTO) fueron fusionados
en [`ESTRUCTURA.md`](ESTRUCTURA.md) durante la reorganización de 2026-07 (PR #27).

## Qué cambió y por qué (con evidencia)

1. **La escala de pre-entrenamiento domina.** ViSpeR (288M, 794 h en español) zero-shot
   le gana por ~20 WER a nuestro mejor fine-tune (ft05, ~19 h de datos):
   45.22 vs 65.05 ([ledger](RESULTS.md), [exp. 02](experiments/02_zeroshot.md)).
   → El modelo de producción pasó a ser **ViSpeR zero-shot (+ LoRA personal opcional)**,
   y el fine-tuning propio quedó como evidencia de investigación.
2. **El scraping masivo de YouTube no es viable** (pared anti-bot; ver
   [exp. 07](experiments/07_datos_y_scraping.md)) → el crecimiento del dataset es
   curado, de a una fuente, con gates de calidad.
3. **El corrector LLM 1-best siempre empeora; el n-best rescoring sí ayuda** (−3.04 WER
   significativo a n=100, [exp. 04](experiments/04_qwen_corrector.md)) → el "corrector"
   del sistema es rescoring, y esa investigación se volvió una línea de trabajo en sí misma.
4. **"Tiempo real" = ventanas por pausas, no streaming causal.** El modelo es
   offline/bidireccional; la demo corta segmentos con VAD visual y logra ~1.1 s por
   segmento ([SPEC](SPEC.md) §4). El streaming causal cuadro a cuadro quedó fuera de scope.
5. **El pre-entrenamiento con currículum ViSpeR-es quedó pausado**
   ([PLAN_CURRICULUM](PLAN_CURRICULUM.md)): costo/beneficio desfavorable frente al punto 1.

## Alcance actual: dos líneas de trabajo

### Research track

**Pregunta:** ¿qué estrategias de uso de LLMs ayudan o perjudican al reconocimiento
visual del habla en español rioplatense, y bajo qué condiciones?

Guía completa de la evidencia: [`RESEARCH.md`](RESEARCH.md). Cubre corrección
1-best, prompts, few-shot, n-best rescoring, oracle, análisis por CER del modelo base,
resultados negativos incluidos, y limitaciones estadísticas.

### Engineering track

**Sistema** local de lectura de labios cerca de tiempo real, demostrable en
vivo: cámara → detección/seguimiento facial → crop → segmentación por pausas visuales →
VSR → rescoring opcional → subtítulos en UI web, con calibración por hablante.

Guía completa: [`SYSTEM_ENGINEERING.md`](SYSTEM_ENGINEERING.md). Arquitectura, latencias
medidas, optimizaciones probadas y descartadas, robustez, y gaps honestos.

## Qué quedó descartado

- Scraping masivo / rotación de IPs (decisión dura, ver CONTRIBUTING.md).
- Full fine-tuning de ViSpeR (overfitea: 61.5 vs 45.2 zero-shot, [exp. 03](experiments/03_visper_finetunes.md)).
- Full-FT para calibración personal (colapsa el 288M, [exp. 10](experiments/10_adaptacion_hablante.md)).
- Corrección LLM 1-best en todas sus variantes de prompt ([exp. 04](experiments/04_qwen_corrector.md)).
- int8 y CTC-greedy en M1 (Pareto-dominados, [exp. 09](experiments/09_velocidad_inferencia.md)).
- Streaming causal real (limitación arquitectural del modelo base).

## Qué sigue abierto

Ver [`FUTURE_WORK.md`](FUTURE_WORK.md). Los grandes: política de datos repo-vs-bucket
(la rama `feature/full-clean-release` propone migrar todo al bucket; main aún versiona
los clips), validación con más hablantes, captura de correcciones humanas editables,
y reducción de latencia del beam search.

## Historia de los datos (resumen)

- **Ronda 1** (~14 h) y **ronda 2** (~19 h): dataset propio de YouTube rioplatense,
  versionado en `data/clips/` + `dataset/` (ver [`DATA_AND_ARTIFACTS.md`](DATA_AND_ARTIFACTS.md)).
- **Release limpio v1** (tag `dataset-clean-v1`, julio 2026): re-construcción + ASR +
  limpieza GPT + discovery de fuentes nuevas, con manifests y reportes en
  `data_pipeline/release/` (portados a esta rama) y datos pesados en bucket. Cerró en la rama
  de PR #25; su integración a main es una decisión abierta.
