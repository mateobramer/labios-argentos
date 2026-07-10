# NEXT_STEPS — próximos pasos por área

Actualizado 2026-07-10 (branch `chore/repo-cleanup-safe-v2`). Complementa
[`TO-DO.md`](../TO-DO.md) (checklist de la rúbrica); acá va la mirada por área con
dependencias explícitas. Estado de cada ítem: nada de esto está empezado salvo
indicación contraria.

## 1. Robustez del producto

- Reinicio automático del `infer_server` si muere (hoy: error por línea en la demo).
- Mensaje guiado cuando la cámara no tiene permiso en macOS.
- `tests/` con pytest para: `norm()` (casos ñ/tildes), WER/CER, dedup de bordes del
  stitching, máquina de estados del VAD (serie sintética), parseo de manifest,
  protocolo CONFIG/READY. (El repo ya tiene tests en `cleaning/visual_quality/`,
  `cleaning/transcript_segmentation/` y `data_discovery/` — el hueco es la demo.)
- `setup.sh` + `run.sh` + `environment.yml` por env (`ptt`, `visper`) — ver TO-DO §3.

## 2. Mejoras de latencia

Hoy: ~1.1 s/segmento (encoder MPS 0.17 s + beam CPU ~0.9 s). El beam es el 80 % del costo.

- Atacar el beam search CPU: batching, poda más agresiva, o decoder alternativo
  (medir contra el sweep de [exp. 09](../experiments/09_velocidad_inferencia.md)).
- Pipeline overlap: inferir el segmento N mientras se captura el N+1 (hoy secuencial).
- Ajuste fino del tope de 4 s y de la pausa de 0.45 s con datos de uso real.
- (Grande, otra arquitectura) VSR causal streaming — hoy fuera de scope.

## 3. Feedback editable y recolección de correcciones

**Gap declarado** ([ENGINEERING_TP](ENGINEERING_TP.md)): hoy existe la donación de
pares clip+texto en `/calibrar`, pero NO la corrección de predicciones en uso real.

- UI: botón "corregir" sobre cada segmento del guion → guarda `(npz, hipótesis,
  corrección)` local, mismo formato que la calibración.
- Eso genera pares `predicción → texto real` para: (a) fine-tune personal continuo,
  (b) entrenar un rescorer supervisado (§5).
- Requiere política de privacidad idéntica a calibración (todo local, opt-in).

## 4. Validación con más hablantes

- Grabar 1-2 personas más (~30 clips c/u) con `build_testset.py` (ya es append+resumible)
  → medir generalización del pipeline de calibración (hoy n=1, [exp. 10](../experiments/10_adaptacion_hablante.md)).
- Replicar el resultado del n-best rescoring (−3.04 WER) en al menos un segundo hablante.

## 5. Mejoras científicas del corrector / n-best

- **Rescorer entrenado** para cerrar la brecha al oracle (26.5 → techo 21.7 a n=100):
  usar los pares de §3 o el self-test como supervisión. Es el ítem de mayor potencial.
- Barrido fino del umbral de CER donde el rescoring empieza a ayudar (hoy: 3 puntos).
- qwen local vs API comercial: costo/latencia/privacidad con números (TO-DO §5).

## 6. Mejoras del modelo visual

- Evaluar bases más nuevas si aparecen (el hallazgo "la escala de pre-entrenamiento
  domina" sugiere que un upgrade de base mueve más que cualquier fine-tune propio).
- Si se retoma el currículum ViSpeR-es: gates go/kill de [PLAN_CURRICULUM](PLAN_CURRICULUM.md).

## 7. Deuda de datos

- **Decisión abierta (equipo): política repo-vs-bucket.** Main versiona clips (~2.9 GB
  en el tree, clone pesado); `feature/full-clean-release` propone todo al bucket con
  repo liviano. Trade-offs: reproducibilidad sin GCP y diff-abilidad (repo) vs clone
  rápido y single-source-of-truth (bucket). Si se adopta el bucket: requiere plan de
  migración + probablemente reescritura de historia (fuera de scope de esta limpieza).
- Decidir el destino de la PR #25: ya se portó su material único liviano; lo que queda
  ahí es la eliminación masiva + manifests grandes. Cerrarla documentando o adaptarla.
- Integrar el release `dataset-clean-v1` al flujo de entrenamiento (hoy los splits
  congelados siguen sobre el dataset original).

## 8. Tareas que requieren GPU, bucket o decisión humana

| Tarea | Requiere |
|---|---|
| Re-entrenar / fine-tunear con el release clean-v1 | GPU (VM L4) + bucket + decisión de splits |
| Migración de datos repo→bucket (§7) | decisión del equipo + bucket |
| Cierre de la PR #25 | decisión del equipo |
| Rescorer entrenado (§5) | GPU chica o incluso CPU + datos de §3 |
| Validación multi-hablante (§4) | personas + tiempo, sin GPU |
| Screenshot/GIF de la demo para el README (TO-DO §1) | cámara + demo corriendo |
