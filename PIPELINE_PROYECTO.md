# Pipeline del proyecto

Lectura de labios en tiempo real para espanol rioplatense.

Este documento ordena el pipeline completo del proyecto: que hacemos, en que orden,
que artefactos produce cada etapa y como sabemos si vale la pena escalar. No es una
guia de instalacion ni un paper; es una guia de trabajo compartida para el equipo.

## 1. Objetivo

Queremos construir un sistema de reconocimiento visual del habla (VSR) para espanol
rioplatense. Dado un video del rostro de una persona hablando, el sistema deberia
producir texto usando solo informacion visual de los labios, sin audio.

El foco rioplatense importa porque los datasets publicos casi no cubren nuestro modo de
hablar: voseo, modismos, ritmo, pronunciacion y temas locales. Por eso el proyecto
combina dos trabajos:

- armar datos propios de calidad;
- medir, paso a paso, que modelo funciona mejor.

## 2. Escalera de avance

| Etapa | Que se hace | Criterio para avanzar |
|---|---|---|
| 1. Prueba minima | Tomar pocos videos, generar clips y revisar que el flujo no rompa. | 10-20 clips mirados a mano; texto y video coinciden. |
| 2. Dataset semilla | Armar un primer conjunto chico pero confiable, con metadata basica y splits iniciales. | Hay variedad minima de hablantes y un test separado. |
| 3. Preparacion visual | Detectar rostro, recortar boca, normalizar fps/resolucion y descartar fallas visuales. | Cada clip queda listo para entrar al modelo VSR. |
| 4. Baseline | Probar un modelo existente sin agregar complejidad propia. | Tenemos un WER inicial y ejemplos de errores. |
| 5. Modelo adaptado | Afinar el modelo fuerte con datos en espanol/rioplatense. | Mejora frente al baseline o muestra donde falta dato. |
| 6. Student causal + KD | Pasar a un modelo liviano para tiempo real y usar destilacion para recuperar precision. | Medimos perdida de WER contra ganancia de latencia. |
| 7. LLM de cierre | Decidir, con el texto parcial acumulado, si ya hay una oracion completa para commitear. | Reduce cortes raros sin depender de audio ni esperar demasiado. |
| 8. Corrector LM | Agregar correccion de lenguaje sobre la salida cruda del VSR. | Mejora texto/WER sin frenar demasiado el sistema. |
| 9. Feedback loop | Permitir correcciones de usuarios y convertirlas en ejemplos revisables. | Las correcciones mejoran dataset, evaluacion o fine-tuning. |
| 10. Cierre | Consolidar la mejor combinacion alcanzable y documentar resultados. | Queda una demo defendible y una comparacion honesta. |

## 3. Datos antes del primer modelo

Esta etapa arma ejemplos de entrenamiento y evaluacion. No resuelve cuando termina una
oracion en tiempo real: eso queda para el modulo de cierre. Aca buscamos datos limpios,
alineados y con metadata confiable.

| Paso | Tarea | Salida esperada |
|---|---|---|
| 1 | Seleccionar fuente | Video elegido, hablante/fuente identificados y calidad visual minima. |
| 2 | Transcribir offline | Texto con timestamps aproximados para construir ejemplos supervisados. |
| 3 | Generar ejemplos | Clips cortos video-texto para entrenar/evaluar; no son la logica final de oracion. |
| 4 | Auditar ejemplos | Muestra revisada: boca visible, texto alineado, un hablante principal. |
| 5 | Armar manifest | Tabla por ejemplo con ruta, texto, fuente, hablante, split y estado de calidad. |
| 6 | Separar test | Test rioplatense fijo que no se usa para decidir fuentes ni ajustar modelos. |
| 7 | Pasar a vision | Solo ejemplos aprobados siguen a crop labial y normalizacion visual. |

### 3.1 Seleccion de fuentes

- Elegir videos donde la boca sea visible buena parte del tiempo: frente o semi frente,
  sin tapa boca, sin edicion excesiva.
- Buscar variedad real: mujeres, varones, edades, registros formales e informales,
  calle, oficios, educacion, salud, politica, humor y vida cotidiana.
- Evitar que el dataset quede dominado por un solo tema o estilo, por ejemplo
  futbol/streaming masculino.
- Registrar desde el principio quien habla, fuente, tema, registro, region aproximada y
  notas de calidad. Si no se sabe algo, se marca como pendiente.

### 3.2 Transcripcion y ejemplos supervisados

- Whisper sirve para obtener texto y tiempos aproximados para armar el dataset offline.
  No es parte del sistema final en tiempo real.
- El texto limpio debe conservar lo util para espanol rioplatense: voseo, lexico local y
  palabras frecuentes del registro real.
- Si el audio tiene musica fuerte, solapamientos o mucha edicion, se marca para revision
  porque puede producir ejemplos malos.
- El objetivo no es tener textos perfectos, sino ejemplos video-texto suficientemente
  alineados para entrenar y evaluar.

### 3.3 Control de calidad y splits

- Revisar muestras por video: texto razonable, boca visible, un solo hablante principal y
  duracion manejable.
- Separar train/valid/test por hablante o fuente cuando sea posible. Si el mismo hablante
  aparece en train y test, la evaluacion queda inflada.
- Mantener un test rioplatense chico y fijo desde temprano. Ese test no se toca para
  elegir videos ni ajustar reglas del modelo.
- El manifest es el puente hacia entrenamiento: sin manifest confiable, el dataset no
  esta listo.

## 4. Preparacion visual para VSR

Despues de tener clips texto-video alineados, todavia falta preparar la senal visual. El
modelo no deberia recibir videos crudos si podemos darle una entrada mas consistente.

| Paso visual | Que hay que resolver |
|---|---|
| Deteccion facial | Ubicar rostro y landmarks en cada frame. Si falla muchas veces, el clip queda marcado. |
| Crop labial | Recortar la region de boca/labios, por ejemplo 96 x 96 px, manteniendo margen suficiente para movimiento. |
| Normalizacion | Unificar fps, tamano, escala de grises/color, duracion y orden de frames. |
| Chunking temporal | Agrupar frames en ventanas que el modelo pueda procesar, cuidando que no rompan la alineacion del texto. |
| Filtro visual | Descartar clips con boca tapada, perfil extremo, rostro fuera de cuadro, multiples caras dominantes o cortes de camara. |
| Salida lista | Guardar clip visual procesado, texto normalizado y metadata. Esto entra al baseline o al entrenamiento. |

En el repo, esta etapa vive en `visual_preprocessing/`. La salida esperada para
entrenamiento es:

```text
data/processed/lip_rois/<titulo>/<clip>.npz
```

En GCS, los ROIs compartidos viven en:

```text
gs://labios-argentos-vsr-data/lip_rois/
```

## 5. Modelos y modulos en tiempo real

En vivo, el sistema no es solo un modelo VSR. El flujo esperado es:

```text
frames de video
-> VSR causal
-> texto parcial
-> LLM de cierre de oracion
-> oracion commiteada
-> corrector LM
-> texto final visible
-> correccion opcional del usuario
-> feedback loop hacia datos/evaluacion/fine-tuning
```

| Modulo | Que cambia | Que miramos |
|---|---|---|
| Baseline | Evaluar un modelo existente con nuestro test rioplatense. | WER inicial, tipos de error y ejemplos concretos. |
| Teacher adaptado | Afinar un modelo fuerte con datos en espanol y luego con nuestro corpus. | Ver si mejora frente al baseline y en que casos falla menos. |
| Student causal | Probar un modelo mas liviano que solo use cuadros pasados. | Medir cuanto empeora WER y cuanto mejora latencia. |
| Destilacion | Entrenar el student usando senales del teacher. | Ver si recupera precision sin perder tiempo real. |
| LLM de cierre | Lee el texto parcial acumulado y responde si ya parece una oracion completa. | Menos cortes prematuros, menos espera innecesaria y latencia aceptable. |
| LLM corrector | Corrige la oracion commiteada sin cambiar el sentido. | Mejora legibilidad/WER sin inventar contenido. |
| Feedback loop | Captura correcciones de usuarios y las convierte en candidatos de dataset. | Mejora sostenida sobre test fijo y sobre casos corregidos. |

### 5.1 LLM de cierre de oracion

Este modulo no usa audio. Mira el texto parcial que el VSR causal viene generando y
decide si conviene commitear una oracion o seguir acumulando.

| Campo | Definicion |
|---|---|
| Entrada | Texto parcial acumulado, ultimos tokens crudos, opcionalmente confianza del VSR y tiempo desde el ultimo commit. |
| Pregunta | Esto ya parece una oracion/frase completa con principio y fin, o falta contexto? |
| Salida | Decision simple: commitear ahora, esperar mas, o marcar baja confianza. |
| Restriccion | Tiene que ser rapido y estable: no puede revisar toda la historia ni bloquear el flujo en vivo. |
| Riesgo | Si commitea temprano, corta ideas; si espera demasiado, sube la latencia y la demo se siente lenta. |

### 5.2 Corrector LM

El corrector LM trabaja sobre una oracion ya commiteada. Su objetivo no es inventar
contenido, sino mejorar legibilidad y corregir errores plausibles del VSR.

Debe preservar:

- significado;
- orden general de la frase;
- marcas rioplatenses utiles, como voseo o lexico local;
- baja latencia.

Debe evitar:

- completar ideas no vistas;
- reemplazar una salida incierta por una frase demasiado fluida pero falsa;
- borrar nombres propios, marcas locales o modismos si son recuperables.

## 6. Componente extra: feedback loop agentico/producto-research

El proyecto no puede quedarse en una cascara de demo. Necesita un componente que mejore
el modelo central o refuerce la investigacion. La pieza propuesta es un feedback loop:
usuarios corrigen transcripciones visuales y esas correcciones se convierten en evidencia
para evaluar, curar datos y eventualmente reentrenar.

### 6.1 Flujo propuesto

```text
1. El VSR produce texto crudo o parcial.
2. El LLM de cierre decide cuando commitear una oracion.
3. El corrector LM propone una version final.
4. El usuario acepta, edita o rechaza esa salida.
5. El sistema guarda un evento de feedback con video, prediccion, correccion y metadata.
6. Una etapa de revision marca el evento como valido, dudoso o descartado.
7. Los eventos validos alimentan un manifest de correcciones.
8. Ese manifest se usa para evaluacion dirigida, curacion del dataset o fine-tuning.
9. Cada nuevo modelo se compara contra el baseline anterior en el mismo test fijo.
```

### 6.2 Que se deberia guardar

Cada correccion deberia generar un registro chico y auditable:

| Campo | Para que sirve |
|---|---|
| `clip_id` o `segment_id` | Vincular la correccion con el video/ROI original. |
| `prediccion_vsr` | Medir el error crudo del modelo visual. |
| `prediccion_corregida_lm` | Medir que cambio propuso el LM. |
| `correccion_usuario` | Capturar el texto considerado correcto por la persona. |
| `decision_usuario` | `accept`, `edit`, `reject` o `unclear`. |
| `confianza_vsr` | Priorizar ejemplos dificiles o inciertos. |
| `latencia_ms` | Controlar si la mejora rompe tiempo real. |
| `fuente`, `hablante`, `split` | Evitar contaminar test y analizar sesgos. |
| `estado_revision` | `pending`, `valid`, `discarded` o `needs_review`. |

Regla importante: una correccion de usuario no entra automaticamente al entrenamiento.
Primero queda como candidato revisable.

### 6.3 Usos del feedback

- **Evaluacion dirigida:** armar un set de casos donde el sistema fallo y medir si una
  nueva version los mejora.
- **Curacion de datos:** detectar fuentes, hablantes o tipos de frase donde el pipeline
  produce errores repetidos.
- **Fine-tuning incremental:** agregar ejemplos validados al entrenamiento sin tocar el
  test fijo.
- **Mejora del corrector LM:** comparar salida cruda, salida corregida y correccion
  humana para medir cuando el LM ayuda o inventa.
- **Investigacion:** documentar si el producto genera datos que realmente mejoran el
  modelo, no solo una interfaz.

### 6.4 Riesgos y controles

| Riesgo | Control |
|---|---|
| Contaminar el test con correcciones vistas durante desarrollo. | Mantener un test fijo separado; las correcciones de test solo se usan para analisis, no para entrenar. |
| Aprender de correcciones malas. | Requiere estado de revision antes de entrar al dataset. |
| Que el LM invente contenido fluido. | Medir hallucination/edit distance y conservar salida cruda para auditoria. |
| Que el feedback sesgue hacia pocos usuarios o temas. | Guardar metadata de fuente/hablante/tema y revisar distribucion. |
| Aumentar latencia en la demo. | Medir latencia del cierre, del corrector y del flujo completo. |

## 7. Estudios a documentar

| Estudio | Pregunta | Comparacion | Resultado |
|---|---|---|---|
| Causalidad y destilacion | Cuanta precision perdemos al pasar de teacher grande a student causal, y cuanto recupera la destilacion? | Teacher vs. student causal vs. student causal con KD. | WER, latencia, real-time factor y memoria. |
| Corrector LM | Un modelo de lenguaje mejora la salida visual cruda sin inventar de mas ni sumar mucha latencia? | VSR solo vs. VSR + LM/corrector. | WER, legibilidad, errores inventados y demora. |
| Criterio de corte | Cortar por oracion/pausa produce mejores datos que usar solo timestamps automaticos? | Whisper directo vs. reglas de oracion/pausa/longitud. | Calidad de alineacion, clips descartados y WER posterior. |
| Feedback loop | Las correcciones de usuarios generan mejoras reales o solo datos ruidosos? | Modelo base vs. modelo ajustado con correcciones validadas. | WER en test fijo, WER en set de feedback, tasa de errores repetidos. |

## 8. Como registrar avances

Cada avance deberia cerrar con una nota corta. No hace falta escribir un informe largo:
alcanza con que otra persona del equipo pueda entender que paso.

Registrar:

- que se probo: dataset, regla de corte, preprocesamiento, modelo o feedback;
- con que datos: fuentes, cantidad de clips, splits y version del manifest;
- que resultado dio: metrica principal y dos o tres ejemplos de error;
- que decision tomamos: seguir, ajustar, juntar mas datos o descartar;
- que falta para el proximo escalon: tarea concreta y responsable tentativo.

## 9. Mapa del repo

| Bloque | Carpeta/archivo | Rol |
|---|---|---|
| Recoleccion | `descargar_procesar.py` | Descarga, transcribe, limpia texto y corta clips alineados. |
| Preprocesamiento visual | `visual_preprocessing/` | Genera ROIs labiales 96x96. |
| Limpieza/curacion | `data_cleaning/` | Audita clips y arma manifests de calidad. |
| Modelos VSR | `vsr_models/` | Splits, fine-tuning y corridas VSR. |
| Evaluacion | `evaluation/` | Baselines, metricas y resultados. |
| Tiempo real futuro | `realtime/` | Buffer causal, cierre de oracion, corrector y feedback de usuario. |

Artefactos sensibles:

- `data/processed/lip_rois/`: ROIs derivados pesados.
- `dataset/`: dataset final curado.
- `vsr_models/runs/`: resultados, logs y checkpoints de corridas.
- `gs://labios-argentos-vsr-data/lip_rois/`: ROIs compartidos.
- `gs://labios-argentos-vsr-data/models/`: modelos/checkpoints compartidos.
