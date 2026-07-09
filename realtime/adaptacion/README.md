# Calibración al hablante (adaptación personal del VSR)

Personaliza el modelo de lectura de labios a **una persona concreta**: graba unas frases,
las convierte en el mismo insumo visual del dataset (ROIs labiales 96×96), y afina el
modelo campeón sobre ellas. Es el lever de WER más grande que encontramos para la demo en
vivo (el post-procesamiento tipo corrector/LM está descartado — ver `evaluation/`).

## Resultado (medido, 2026-07-03)

Test personal de 15 frases retenidas, WER (±IC por ser solo 15 clips):

| modelo | WER personal |
|---|---|
| FT06 sin adaptar | 66% |
| **FT06 + 96 frases (adaptado)** | **55%**  (−11 pts) |
| FT06 + 30 frases (random) | 57%  (recupera ~78% del beneficio) |
| ft05 + 96 frases (adapt01, base anterior) | 54% |

**Aprendizajes clave:**
- La adaptación baja **~10-11 pts** y **no depende de la base** (ft05 y ft06 adaptados
  terminan casi igual, ~54-55): la adaptación domina, el modelo general casi no importa
  para el resultado personal.
- **Con ~30 frases alcanza** para la mayor parte de la mejora → menos fricción para el
  usuario de la demo (no hacen falta 120).
- **Elegir frases por "dificultad" del modelo NO ayudó** (señal de selección casi nula:
  las vocales, que son el error dominante, están en todas las frases). La versión que sí
  promete es un **loop de dos pasos**: grabar poco → ver dónde falla el modelo *en esa
  persona* → generar frases que ataquen *eso* → grabar esas. **Pendiente** (ver más abajo).

## Flujo de punta a punta

```
1. grabar        realtime/src/grabar_server.py     -> grabaciones/frase_NNN.mp4 + .txt
2. recorte labial  src/preproc_grabaciones.py       -> rois/frase_NNN.npz  (T,96,96) @25fps
3. splits        src/armar_splits_personal.py       -> splits + export para eval
4. adaptar+eval  src/run_adaptacion.sh              -> modelo adaptado + WER personal
```

### 1. Grabar (local, navegador)
```bash
# env realtime (ver ../requirements.txt). Sirve las 120 frases de frases_grabacion.md.
python -m realtime.src.grabar_server        # abrir http://localhost:8001
```
Push-to-talk: mantené apretado, leé la frase, soltá. Reanudable (salta a la primera sin
grabar). Guarda pares `mp4`+`txt` normalizados en `grabaciones/`.

### 2. Recorte labial (local, CPU)
```bash
python -m realtime.adaptacion.src.preproc_grabaciones
```
Reusa el mismo pipeline visual del kiosko (`realtime/src/preprocess_live.py`): MediaPipe →
warp a cara media → gris → crop 96×96 → 25 fps. Descarta clips sin cara estable (<80%).

### 3. Splits personales
```bash
python -m realtime.adaptacion.src.armar_splits_personal
```
Arma train/val/test personal (por defecto 96/9/15) y un export en formato del evaluador de
Gimeno para medir el WER personal antes/después.

### 4. Adaptar y evaluar (GPU)
```bash
# en la VM (env vsr-factors). Afina el modelo base con las frases y evalúa en el test personal.
bash realtime/adaptacion/src/run_adaptacion.sh
```
Es un fine-tune CONTINUADO desde el checkpoint base (no desde cero): `lr 1e-5`, augment,
early stopping. ~15 min de L4 por adapter.

## Datos y artefactos

- **Grabaciones (`grabaciones/`) y ROIs (`rois/`) NO se versionan** (privacidad + tamaño;
  ver `.gitignore`). Cada persona graba las suyas.
- **Frases** (`frases_grabacion.md`): 120 frases pensadas para cubrir variedad de
  movimientos de boca. Se pueden editar/ampliar.
- **Modelos base y adaptados** (en el bucket del proyecto):
  - base campeón: `gs://labios-argentos-vsr-data/models/ft06/best.pth`
  - adaptado a Joaco (ejemplo): `gs://labios-argentos-vsr-data/adaptacion/adapt02_best.pth`
- La receta de fine-tune reusa `vsr_models/src/fine_tune.py` (está en `main`).

## Para seguir (ideas ordenadas por valor)

1. **Loop activo de 2 pasos** (la versión buena de "menos frases pero dirigidas"): grabar
   ~20 frases → correr el modelo sobre la persona → detectar sus palabras/sonidos peores →
   generar frases dirigidas a ESO → grabar esas. Requiere una segunda tanda de grabación.
2. **Bajar el mínimo de frases**: barrer 10/20/30 para el punto de fricción mínimo usable.
3. **Wire del kiosko al modelo adaptado**: la demo carga `adapt*_best.pth` en vez del base.
4. **VAD visual** para sacar el push-to-talk (apertura bucal con los landmarks de MediaPipe).
