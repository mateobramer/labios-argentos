# ESTADO ACTUAL — handoff para retomar (2026-06-29)

Documento de compactación: todo lo necesario para continuar sin la conversación.

## Dónde estamos
- **Fase A (procesar 39 fuentes nuevas): COMPLETA.** 3244 npz curados de 30 fuentes (informal
  rioplatense), 102 clips de música filtrados. Resumen en `full-run/RESULTADO.md`.
- **Fase B (entrenar ft03/ft04): EN MARCHA.** GPU conseguida 2026-06-29 ~10:31 UTC.
  - VM **`labios-vsr-train`** RUNNING en **us-central1-b** (L4), creada desde el snapshot. STARTEPOCH=1782729108.
  - **FT03 (config v1 + data nueva): COMPLETO Y EVALUADO.** Monitor FT04 en vivo: **`bne4tut2x`**. Watchdog cron: **`b11e5c5d`** (:42). (IDs viejos muertos: `b8pbitiqj`/`bl8493a17`/`bpufgtglx`, crons `464af909`/`e6be546c`.)
  - **PHASE_SPLITS_DONE OK: train=8067, val=466, test=658** (test/val congelados = v1/v2; train +3249 = data nueva). Ablación válida.

## RESULTADOS
| modelo | config | datos train | clips test | %WER | %CER |
|---|---|---|---|---|---|
| **v1** (ft01) | full FT | viejos (4818) | **149** | 75.173 ± 2.231 | 45.148 ± 1.744 |
| **ft03** | = v1 | viejos+nuevos (8067) | **658** | 68.934 ± 1.266 | 41.012 ± 0.956 |
| **ft04** | v2 (freeze+aug)* | viejos+nuevos (8067) | 658 | **69.730 ± 1.298** | 42.286 ± 0.949 |
| **v2** (ft02) | freeze+aug | viejos | ? | _FALTA (ckpt en bucket inaccesible)_ | |

- **ft03 vs ft04 (ambos @658, comparables): 68.93 vs 69.73 → diferencia NO significativa (IC se solapan).** Con esta cantidad de datos, congelar el frontend + augment (config v2) NO mejoró frente al full fine-tuning (config v1); si algo, marginalmente peor. El full FT (ft03) es el mejor modelo nuevo.
- Cierre OK (2026-06-30 ~02:35Z): resultados ft04 en full-run/train/, VMs labios-vsr-eval y labios-vsr-train BORRADAS, cron+monitor cerrados. Modelo ft04 preservado en snapshot `labios-ft04-20260629`. **Sin VMs gastando.**

- **OJO comparación: NO es head-to-head válido todavía.** ft03 se evaluó sobre el test COMPLETO (658 clips); el número de v1 que tengo (`ft01.wer` en scratchpad) se calculó sobre solo **149 clips** (subconjunto de las MISMAS 2 fuentes de test congeladas — confirmado: su `test.inf` tiene 149 líneas; IC ±2.23 vs ±1.27 de ft03). Mismas *fuentes*, distinto *N de clips*. El 75.17 vs 68.93 NO permite concluir "mejora significativa" (afirmación previa ERRÓNEA, corregida).
- **Para comparar bien hay que re-evaluar v1 y v2 sobre los mismos 658** → requiere sus `best.pth`, que están en el bucket inaccesible. Bloqueado hasta conseguir acceso / que el compañero los pase.
- **Sí es válido**: ft03=68.93@658 como número absoluto, y ft03 vs ft04 (ambos @658) entre sí.
- *FT04 = "config v2" RECONSTRUIDA: el usuario eligió `freeze frontend` (60 params, inequívoco) + augment estándar VSR (`RandomCrop(88)+HFlip(0.5)` en train; val/test = CenterCrop). La receta de augment original de v2 vivía en el train.log de ft02 (bucket inaccesible) → ft04 NO es bit-idéntico a v2; caveat a documentar en RESULTADO.md.
- **`fine_tune.py` parcheado** para soportar `--freeze`/`--augment` (no existían). Copia en repo local `vsr_models/src/fine_tune.py` (antes el módulo vivía SOLO en el snapshot) y en la VM (backup `.bak`). Smoke v2 OK.
- **FT04 ENTRENADO** (early-stop ep3, best.pth 210MB). **EVAL cortada al 51% por el auto-stop de 12h** (VM start 10:29Z → max-run-duration 43200s → STOP 22:29Z, mid-eval).
  - **VM TERMINATED**; disco persistente en us-central1-b CON ft04/best.pth. **Snapshot de seguridad `labios-ft04-20260629`** (incluye ft04/best.pth + env + 9212 npz + export del test).
  - **Recuperación en curso (2026-06-29 22:35Z):** bucle reintentando `gcloud compute instances start` en us-central1-b (stockout intermitente; el disco es zonal en b). Cuando arranque → correr SOLO la eval con `eval_ft04_only.sh` (scratchpad) que reusa best.pth + el export (~/data/Rioplatense). NO re-entrenar. Si b no libera capacidad, crear VM nueva desde `labios-ft04-20260629` en otra zona y correr eval ahí.
  - **OJO: NO usar autograb_train.sh para esto** — usa el snapshot VIEJO (sin ft04) y re-entrenaría todo.
  - **2026-06-29 23:47Z: stockout TOTAL de GPU en us-central1 (L4+T4, a/b/c)** + brevemente en us-east1/us-west1. No hay capacidad. labios-vsr-eval se creó y se auto-borró por creates interrumpidos (no quedó ninguna VM prendida — costo OK).
  - **Recuperación AUTÓNOMA vía cron `abf8f2da`** (cada ~23 min): reintenta `create_eval_vm3.sh` (consigue GPU desde el snapshot ft04 en us-central1 a/b/c L4→T4 + us-east1/us-west1 L4, lanza `eval_ft04_only.sh`) hasta conseguir GPU; cuando ve `ALL_DONE_FT04` hace el CIERRE (scp test.wer/.inf de ft04 → full-run/train/, CSV por longitud, tabla v1/ft03/ft04, actualizar artifact, STOP+DELETE VM, borrar cron, avisar al usuario). Scripts en el scratchpad de la sesión.
  - **2026-06-29 23:57Z: GPU conseguida — L4 en us-east1-b** (us-central1 seguía seco; us-east1 tenía capacidad y cuota OK). VM `labios-vsr-eval` RUNNING us-east1-b, best.pth verificado (210MB), eval-only corriendo (PID 1325, `~/eval_ft04_run.log`). ETA ~2.7h → cierre ~02:40Z. Monitor en vivo **`bhc94ocjt`** (us-east1-b) + cron respaldo `abf8f2da`.
  - **Costo: quedan $26.10 (confirmado por el usuario en consola, 2026-06-30 ~00:00Z) → ~$23.90 gastados.** Falta solo la eval (~$2.5) → cierre con ~$23-24 disponibles. El usuario pidió dejarlo correr y avisar SOLO al cierre.
  - **2026-06-29 11:02 — FATAL #1 en PHASE_SETUP: `unzip: command not found`** (no venía en la imagen base). Fix: instalé `unzip`, borré `~/zenodo/extracted` vacío, relancé. Agregué install de `unzip` defensivo a `train_orchestrator.sh`.
  - **2026-06-29 11:09 — FATAL #2 en PHASE_DATA: rsync GCS 403** (bucket inaccesible, ver Infra). Fix: subí los 5968 npz viejos por scp local→VM (tar 4.1G en 3 trozos por límite de tiempo del harness), 9212 npz totales en disco. Parcheé PHASE_DATA para saltear el rsync si ya hay ≥8000 npz. Relancé (PID 6224) y pasó SPLITS OK.
  - Auto-grab ya cumplió (no hace falta re-lanzarlo salvo que la VM muera).

## Qué está corriendo AHORA
- **Auto-grab de GPU** (background): `bash new-data-fine-tuning/scripts/autograb_train.sh`.
  Reintenta L4→T4 en us-central1 a/b/c, 8 rondas (~24 min) por corrida. Al conseguir GPU crea
  la VM **`labios-vsr-train` DESDE el snapshot** y lanza `train_orchestrator.sh`.
  **Si termina sin GPU (SIN_CAPACIDAD): hay que RE-LANZARLO** (lo hace el watchdog, o a mano).
- **Watchdog cron `464af909`** (cada hora :23, session-only): chequea estado, re-lanza auto-grab
  si hace falta, y al terminar/fallar/colgarse actúa (snapshot+stop / stop+avisar / reset+resume).
- **Monitor en vivo**: se engancha recién cuando la VM existe (hay que saber la zona). Comando:
  ssh a `labios-vsr-train`, seguir `~/train.log`, emitir PHASE_*/FTxx_DONE/EVAL_*/ALL_DONE/FATAL +
  alerta si no responde 3 polls.

## Infra (GCP, proyecto visual-speech-recognition-nlp, fgutman@udesa.edu.ar)
- **Snapshot Fase A: `labios-full-20260629-0429`** (40 GB: repo + venvs proc/visual + 3244 npz nuevos
  + manifests). La VM de entrenamiento se crea de acá.
- **VM `labios-vsr-full`: TERMINATED** (datos redundantes con el snapshot; se puede borrar para ahorrar ~$0.40/día).
- **GCS `gs://labios-argentos-vsr-data/`**: `lip_rois/` (npz VIEJOS del dataset original), `models/{ft01_v1,ft02_v2}/` (checkpoints v1/v2 + train.log + eval).
- **OJO: el bucket es de OTRO proyecto (de un compañero) y NO tenemos acceso — ni lectura ni escritura.** Lo confirmé el 2026-06-29: tanto la service account de la VM (`303191263643-compute@...`) como mi cuenta local (`fgutman@udesa.edu.ar`) reciben 403 `storage.objects.get` sobre `lip_rois/`. No puedo arreglarlo con IAM (no soy admin de ese bucket).
  - **Consecuencia 1 (PHASE_DATA):** los npz VIEJOS (5968, dataset original, 4.2G) NO se pueden bajar de GCS. Están LOCAL en `data/processed/lip_rois`. Solución aplicada: tar local → scp a la VM → extraer en `~/labios-argentos/data/processed/`. `train_orchestrator.sh` PHASE_DATA ya patcheado: si hay ≥8000 npz saltea el rsync; rsync queda como fallback no-fatal.
  - **Consecuencia 2 (EVAL):** la EVAL baja `models/ft01_v1/best.pth` y `ft02_v2/best.pth` de GCS para re-evaluarlos sobre test 658 → VA A FALLAR (best-effort, no corta). Para la tabla comparativa v1/v2 vs ft03/ft04 hay que conseguir los WER de v1/v2 de otra fuente. **v1 YA lo tengo registrado: %WER 75.173 ± 2.231, %CER 45.148 ± 1.744** sobre test 658 (estaba en scratchpad `ft01.wer`). FALTA el de v2 (buscar en resultados locales / pedírselo al compañero). ft03/ft04 sí dan WER absoluto sobre el mismo test 658.
- **Stockout de GPU intermitente** (L4 y T4) → por eso el auto-grab reintenta.

## El experimento (ablación de datos)
Mismo test/val que v1/v2; train suma las 3244 nuevas. Condiciones: datos nuevos SOLO a train,
val/test congelados (`armar_splits` lo garantiza), speaker-independent, ft03=config v1, ft04=config v2,
eval de los 4 sobre test 658. Detalle y caveats en `PLAN_ENTRENAMIENTO.md`.

Configs exactas:
- **ft03 (v1, full FT):** `--lr 1e-4 --batch 1 --accum 8 --max-frames 400 --paciencia 5`
- **ft04 (v2, frozen+aug):** idem `+ --freeze frontend --augment`
- Checkpoint base: `~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth`; config: `~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml`
- Eval: `exportar_para_gimeno` del test (2 fuentes, sin cap) → `vsr_main.py --database Rioplatense --scenario zero-shot --load-vsr <ckpt>`.

## Pipeline de entrenamiento (en `train_orchestrator.sh`, corre en la VM)
PHASE_SETUP (setup_modelo_gimeno.sh: env vsr-factors + Zenodo 8.5GB) → PHASE_DATA (rsync npz viejos
de GCS) → PHASE_SPLITS (armar_splits, valida test≈658) → SMOKE (fine_tune --smoke; si falla NO entrena)
→ FT03 → FT04 → EVAL (best-effort, 4 ckpts) → ALL_DONE. Marcadores en `~/train.log`.

## Cierre esperado (lo hace el watchdog al ver ALL_DONE)
snapshot del disco → scp de `train.log`, `runs/ft0{3,4}/*.{wer,inf}`, `*train.log`, `splits.log`,
`eval_*.log` a `full-run/train/` → STOP VM → borrar cron+Monitor → tabla comparativa v1/v2/ft03/ft04
con `.claude/skills/resultados/metricas.py`.

## Costo
Gastado ~$11 de $47.44. Entrenamiento estimado ~$3-6. VM con max-run-duration 12h (auto-stop).

## Scripts (durables, en `new-data-fine-tuning/scripts/`)
`autograb_train.sh` (auto-grab+lanzar), `train_orchestrator.sh` (pipeline en VM),
`setup_and_run.sh` + `filtro_musica.py` (Fase A, ya usados), `autograb_launch.sh` (Fase A).

## Pendientes / por si hay que recuperar
- 5 fuentes de Fase A cayeron por bot-block/age-gate de YouTube (recuperables con cookies/otra IP):
  `aQlIHv_K0zk`, `ZhPgRWjBWvk`, `scQ7nPWsA8g`, `zyr7wpiIt18`, `wB4JpMNFqb4`.
- Si se pierde TODO el estado de sesión: re-lanzar `autograb_train.sh`, y re-crear el cron watchdog
  y el Monitor con la lógica de arriba. Los scripts y este doc bastan para reconstruir.
