# Plan de entrenamiento — ft03 / ft04 (ablación de datos)

## Estado actual (compact)
- **Fase A COMPLETA** (2026-06-29): 3244 npz nuevos curados de 30 fuentes (informal rioplatense),
  102 clips de música filtrados. 5 fuentes caídas por bot-block de YouTube (recuperables).
- Datos nuevos en: **snapshot `labios-full-20260629-0429`** (40 GB: repo + venvs + npz nuevos + manifests).
- Datos viejos (dataset original, ~5968 npz / test+val+train de v1/v2): en **GCS `gs://labios-argentos-vsr-data/lip_rois/`**.
- Checkpoints v1/v2 + sus train.log/eval en `gs://.../models/{ft01_v1,ft02_v2}/`.
- Presupuesto: ~$36 restante de $47.44 (gastado ~$11).

## El experimento (ablación de datos, validado con el usuario)
Misma arquitectura, mismos hiperparámetros, **mismo test** que v1/v2; lo único que cambia es
que el **train** suma las 3244 clips nuevas. Condiciones acordadas:
1. **Datos nuevos → SOLO train.** val y test quedan **idénticos** a v1/v2 (no mover nada a val).
   `armar_splits` ya lo garantiza: test/val hardcodeados, el resto (viejo train + nuevo) → train.
2. **Speaker-independent:** ninguna fuente nueva pisa val/test (verificado).
3. **ft03 = config de v1** (full FT): `--lr 1e-4 --accum 8 --max-frames 400 --paciencia 5 --batch 1`.
   **ft04 = config de v2** (frozen+aug): idem **+ `--freeze frontend --augment`**.
4. **Eval de los 4 (v1, v2, ft03, ft04) sobre el MISMO test completo (658)**, mismo beam, para IC comparables.

### Caveats honestos
- Etiquetas de train mixtas (nuevo=large-v2, viejo=large-v3): leve, no afecta la métrica de test.
- Test chico (658) → solo mejoras grandes serán concluyentes (IC ~±2.2; usar helper `sig`).
- Config fija mide "efecto de los datos", no el mejor modelo posible (no se re-tunea).

## Ejecución autónoma (mientras el usuario trabaja)
Auto-grab de GPU (L4→T4, us-central1 a/b/c, reintento hasta conseguir) → al caer, crea VM
**desde el snapshot** (trae repo+venvs+npz nuevos) y corre `train_orchestrator.sh`:

1. **PHASE_SETUP**: `setup_modelo_gimeno.sh` (env conda `vsr-factors` + Zenodo 8.5 GB + repo Gimeno + parches).
2. **PHASE_DATA**: `gcloud storage rsync` de los npz viejos de GCS → `data/processed/lip_rois/` (junto a los nuevos).
3. **PHASE_SPLITS**: `armar_splits` → verifica train↑, val/test sin cambios, speaker-independent OK.
4. **SMOKE**: `fine_tune --smoke` (1 batch) → **si falla, NO entrena** (corta, no quema GPU).
5. **FT03 / FT04**: entrena ambos (configs de arriba) → `best.pth` cada uno.
6. **EVAL (best-effort)**: export test (658) → `vsr_main.py --load-vsr <ckpt>` para ft03, ft04, v1, v2 → `test.wer/.inf`.

### Seguridad de costo y fallas
- VM con **max-run-duration = 12h** (auto-stop, tope duro de costo).
- Marcadores de fase en el log → Monitor en vivo (alerta de no-respuesta/cuelgue) + watchdog horario.
- Ante **cuelgue**: reset + resume. Ante **fallo de stage crítico** (setup/splits/smoke): el watchdog
  apaga la VM y avisa (no se sigue quemando).
- Al terminar: **snapshot** (preserva checkpoints+eval) + pull de logs/.wer + **stop VM**.
- Pendiente menor: arreglar permiso GCS del SA para subir checkpoints (hoy se preservan vía snapshot).

## Artefactos esperados
- `vsr_models/runs/ft03/`, `ft04/`: `best.pth`, `train.log`, eval `.inf`/`.wer`.
- Tabla comparativa v1/v2/ft03/ft04 (WER/CER ± IC, significancia) → `RESULTADO.md` (skill `/resultados`).
