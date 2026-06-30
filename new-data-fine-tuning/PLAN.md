# new-data-fine-tuning — ablación de datos (ft03 / ft04)

## Hipótesis

Sumar ~10.7 h de habla rioplatense informal **in-domain** (storytime/monólogo a cámara,
ver `claude-videos/candidatos.csv`) al train mejora el WER en test, manteniendo
arquitectura, hiperparámetros y test **idénticos** a v1/v2.

Es una **ablación de datos**: la única variable que cambia es el train. Cualquier cambio de
WER es atribuible a los datos, no al modelo.

## Por qué la comparación es válida

- Los splits son **speaker-independent y hardcodeados por fuente** (`vsr_models/src/armar_splits.py`).
- **test** = 2 fuentes (LE DIJE SOY ARGENTINO, ME ACUSARON DE BRUJA) — ambas storytime informal.
- **val** = 3 fuentes (Azzaro Racing, Charla del amor/Coscu, Ginecólogos Telefe).
- Las 39 fuentes nuevas **no** están en esas listas → caen automáticamente en **train**.
  val y test no se tocan ⇒ comparable contra v1/v2.
- Verificado: ninguno de los 15 hablantes nuevos coincide con las fuentes de val/test (sin fuga).

Además, como test es storytime informal, los datos nuevos son in-domain: el escenario más
favorable para que el aporte se note.

## Diseño de los dos runs

| Run | = config de | Estrategia | Flags clave |
|---|---|---|---|
| **ft03** | v1 (ft01) | full FT, sin congelar, sin augment | `--lr 1e-4 --accum 8 --max-frames 400 --paciencia 5` |
| **ft04** | v2 (ft02) | frontend congelado + augmentation | `--freeze frontend --augment --lr 1e-4 --accum 8 --max-frames 400 --paciencia 5` |

- Mismo checkpoint de partida que v1/v2: `vsr-liprtve-si.pth` (Gimeno).
- **El resto de los flags se copian del `train.log` de ft01/ft02** para garantizar config idéntica;
  lo único que cambia entre {ft01→ft03} y {ft02→ft04} es el dataset.

## Evaluación

- Eval con las mismas settings (beam 10, GPU) **sobre el mismo test**.
- Recomendado: evaluar los 4 (v1, v2, ft03, ft04) sobre el **test completo (658)** —el "próximo"
  que ya pedían ambos RESULTADOS— para IC más ajustados. v1/v2 pueden re-evaluarse desde sus
  `best.pth` en GCS sin reentrenar.
- Veredicto por solapamiento de IC con el helper `metricas.py` del skill `/resultados`.

## Secuencia (el orden importa)

1. **Procesar los 39 videos** (gate 0 ya hecho = `claude-videos/candidatos.csv`). Por cada uno,
   vía skill `/nueva-fuente`:
   - `python descargar_procesar.py "URL"`  (Whisper large-v3 para transcripción final)
   - gate 1: `python -m data_cleaning.src.auditar_alineacion "<titulo>"`  ← protege clip↔texto
   - `python -m visual_preprocessing.src.preprocesar "<titulo>"`  (env MediaPipe)
   - `python -m data_cleaning.src.detectar_clips_malos "<titulo>"` → revisar → `--materializar`
   - registrar fila en `data/metadata/fuentes.csv`
2. **Re-armar splits:** `python -m vsr_models.src.armar_splits` (las nuevas fuentes entran a train).
   Confirmar que val/test siguen iguales y el chequeo speaker-independent pasa.
3. **Entrenar ft03 y ft04** en la VM L4 (env `vsr-factors`).
4. **Evaluar** ft03, ft04 (y re-evaluar v1, v2) sobre test 658.
5. **RESULTADOS.md** de cada run con el skill `/resultados` (plantilla de la casa) + tabla
   comparativa v1/v2/ft03/ft04.

## Realidades / riesgos

- **Cuello de botella = procesamiento de datos**, no el entrenamiento (con ~13h, v1/v2 cortaron
  por overfitting en epoch 3; los runs son cortos).
- **~40% de descarte esperado en el preproc** (cara no frontal). 10.7h crudas → ~6h curadas estimadas.
  Por eso `candidatos.csv` sobre-aprovisiona arriba de 10h.
- **Compute:** preproc MediaPipe corre local; Whisper y training quieren la VM (GPU). El training
  no corre en la Mac.
- **Presupuesto GCP edu acotado (~$50):** costo estimado modesto (pocas GPU-horas), pero a tener en cuenta.
- **No editar a mano** lo generado bajo `data/` ni `dataset/`. Pesos → GCS, no al repo.

## Estado

- [x] Gate 0: selección verificada (`claude-videos/candidatos.csv`, 39 videos / ~10.7h)
- [ ] Procesamiento de datos (etapas 1–4)
- [ ] Re-armar splits
- [ ] ft03 (entrenamiento)
- [ ] ft04 (entrenamiento)
- [ ] Evaluación sobre test 658 + RESULTADOS
