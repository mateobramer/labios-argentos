# Estructura del proyecto

Esta guía define cómo organizar el código nuevo del proyecto. La idea es evitar scripts,
notebooks, datos y resultados desparramados por la raíz.

## Regla general

- La raíz queda para archivos generales: `README.md`, `requirements.txt`, `AGENTS.md`,
  `CLAUDE.md`, documentación general y entrypoints principales.
- Cada bloque del sistema vive en su propia carpeta.
- Dentro de cada bloque, la lógica reutilizable va en `src/`.
- Los notebooks documentan experimentos, pero no deberían ser el único lugar donde vive
  la lógica importante.
- Los datos generados grandes, pesos y corridas pesadas no se suben sin revisar primero
  `.gitignore` y el tamaño del commit.

## Estructura de un módulo

Cuando se cree un módulo nuevo, usar esta forma base:

```text
nombre_modulo/
  README.md              # qué hace el módulo y cómo se conecta con el resto
  notebooks/             # experimentos numerados y reproducibles
    01_exploracion.ipynb
    02_experimento.ipynb
  src/                   # funciones reutilizables
    __init__.py
    ...
  data/                  # solo si son datos propios de ese módulo
    raw/
    processed/
  outputs/               # figuras, tablas, predicciones o ejemplos chicos
  runs/                  # corridas locales; no subir si son pesadas
```

No hace falta crear todas las carpetas desde el inicio. Se crean cuando el módulo las
necesita.

## Módulos esperados

Estos son los bloques naturales del proyecto. Los nombres pueden ajustarse, pero la
separación debería mantenerse.

### `data_pipeline/`

Recolección y armado del dataset.

Va acá:

- descarga de fuentes;
- transcripción offline;
- generación de corpus;
- armado de clips supervisados;
- metadata de fuentes y hablantes;
- controles de calidad del dataset;
- scripts para actualizar `data/metadata/`.

No va acá:

- crop labial;
- entrenamiento de modelos;
- lógica de inferencia en tiempo real.

Nota: `descargar_procesar.py` hoy vive en la raíz como script principal. Si crece, la
lógica debería migrar gradualmente a `data_pipeline/src/` y dejar el archivo de raíz como
entrypoint fino.

### `visual_preprocessing/`

Preparación visual para VSR.

Va acá:

- detección de rostro;
- landmarks faciales;
- recorte de boca/labios;
- normalización de fps, tamaño, color o escala de grises;
- filtros de calidad visual;
- generación de entradas visuales listas para el modelo.

### `vsr_models/`

Modelos de lectura de labios.

Va acá:

- baseline;
- teacher adaptado;
- student causal;
- destilación;
- entrenamiento, validación e inferencia offline;
- configuración de modelos.

### `realtime/`

Sistema en tiempo real.

Va acá:

- buffer de frames;
- inferencia causal por chunks;
- texto parcial o especulativo;
- LLM de cierre de oración;
- LLM corrector;
- demo o display en vivo.

El LLM de cierre de oración no pertenece al preprocesamiento de datos. Su trabajo es
mirar el texto parcial que viene produciendo el VSR en tiempo real y decidir si ya parece
una oración completa para commitear.

### `evaluation/`

Métricas, comparaciones y reportes.

Va acá:

- WER;
- latencia;
- real-time factor;
- análisis de errores;
- comparación entre baseline, teacher, student, destilación y LLMs;
- tablas y figuras finales.

## Notebooks

Los notebooks sirven para explorar y mostrar experimentos, pero deben llamar funciones de
`src/` cuando algo se repite o se vuelve importante.

Convención:

```text
notebooks/
  01_nombre_corto.ipynb
  02_nombre_corto.ipynb
  03_resultados.ipynb
```

Cada notebook debería tener al principio:

- objetivo del experimento;
- datos usados;
- versión o commit aproximado;
- resultado esperado o métrica a mirar.

## Datos y resultados

El dataset principal sigue viviendo en `data/`:

```text
data/
  videos/       # videos crudos; no subir nuevos por accidente
  corpus/       # transcripciones y corpus
  clips/        # clips alineados video-texto
  metadata/     # inventario de fuentes, hablantes y notas
```

Si un módulo necesita datos propios, puede tener `modulo/data/`, pero solo para datos
derivados o auxiliares de ese módulo. Evitar duplicar el dataset completo en varias
carpetas.

Antes de commitear:

- no subir videos crudos nuevos;
- no subir pesos grandes;
- no subir corridas completas de entrenamiento;
- revisar `git status` y confirmar que el commit contiene solo lo esperado.

## Cómo decidir dónde poner algo

- Si descarga, transcribe o arma clips: `data_pipeline/`.
- Si recorta labios o prepara frames: `visual_preprocessing/`.
- Si entrena o evalúa modelos VSR offline: `vsr_models/`.
- Si corre en vivo o decide cuándo commitear texto: `realtime/`.
- Si calcula métricas o arma tablas de resultados: `evaluation/`.
- Si es documentación general: raíz o `docs/`.

Si algo no encaja en ningún módulo, primero discutirlo antes de crear otra carpeta.
