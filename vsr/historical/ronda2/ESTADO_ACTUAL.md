# Cierre de campaña — ft03/ft04 (2026-06-29/30)

Registro histórico del cierre de la fase B de la ronda 2: entrenamiento y evaluación
de ft03 y ft04. Snapshot final del estado en el momento del cierre — no refleja el
estado actual del repo.

## Resultado

- **Fase A** (procesar 39 fuentes nuevas): completa. 3244 npz curados de 30 fuentes,
  102 clips de música filtrados. Detalle en `full-run/RESULTADO.md`.
- **Fase B** (entrenar ft03/ft04): completa. Splits: train=8067 (viejos+3249 nuevos),
  val=466, test=658 (test/val congelados, ablación válida).

| modelo | config | datos train | clips test | %WER | %CER |
|---|---|---|---|---|---|
| **v1** (ft01) | full FT | viejos (4818) | 149 | 75.173 ± 2.231 | 45.148 ± 1.744 |
| **ft03** | = v1 | viejos+nuevos (8067) | 658 | 68.934 ± 1.266 | 41.012 ± 0.956 |
| **ft04** | v2 (freeze+aug) | viejos+nuevos (8067) | 658 | 69.730 ± 1.298 | 42.286 ± 0.949 |
| **v2** (ft02) | freeze+aug | viejos | — | no re-evaluado (checkpoint inaccesible) | |

- **ft03 vs ft04 (ambos @658, comparables): 68.93 vs 69.73 — diferencia no significativa**
  (IC se solapan). Congelar el frontend + augment (config v2) no mejoró frente al full
  fine-tuning (config v1); el full FT (ft03) es el mejor modelo nuevo de esta ronda.
- **Comparación con v1/v2 no es head-to-head estricta**: v1 se evaluó históricamente
  sobre un subconjunto de 149 clips (2 fuentes), no sobre los 658 completos; el
  `best.pth` de v2 quedó en un bucket de otro proyecto sin permisos de acceso, así que
  no se pudo re-evaluar sobre el mismo test. El número de v1 (75.17 WER) es válido como
  referencia absoluta, pero la comparación directa v1 vs ft03 no permite concluir
  "mejora significativa" sin re-evaluar ambos sobre el mismo N.
- **FT04 = config v2 reconstruida** (no bit-idéntica a v2 original): freeze de frontend
  + augment estándar (`RandomCrop(88)+HFlip(0.5)` en train; val/test = CenterCrop). La
  receta original de augment de v2 vivía en un log inaccesible en ese mismo bucket.
  Caveat documentado en `full-run/RESULTADO.md`.
- `fine_tune.py` se extendió con soporte para `--freeze`/`--augment` (no existían antes).

## Infraestructura y costos

Proyecto GCP `visual-speech-recognition-nlp`. Entrenamiento en VM L4 spot
(`labios-vsr-train`), con recuperación automática ante stockouts de GPU intermitentes
en la región. Snapshot de seguridad del checkpoint final: `labios-ft04-20260629`.
Costo total de la campaña: ~$24 de $47.44 presupuestados. Sin VMs activas al cierre.

**Limitación de acceso documentada**: los checkpoints de v1/v2 (`ft01_v1`, `ft02_v2`)
viven en un bucket de otro proyecto sin permisos de lectura para esta cuenta —
bloqueó la re-evaluación de v1/v2 sobre el test completo de 658 clips.

## Experimento (ablación de datos)

Mismo test/val que v1/v2; train suma las 3244 fuentes nuevas de la ronda 2.
Condiciones: datos nuevos solo a train, val/test congelados, speaker-independent,
ft03=config v1, ft04=config v2, eval de los 4 sobre test 658. Detalle y caveats en
`PLAN_ENTRENAMIENTO.md`.

Configs exactas:
- **ft03 (v1, full FT):** `--lr 1e-4 --batch 1 --accum 8 --max-frames 400 --paciencia 5`
- **ft04 (v2, frozen+aug):** idem `+ --freeze frontend --augment`
- Checkpoint base: `~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth`; config:
  `~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml`
- Eval: `exportar_para_gimeno` del test (2 fuentes, sin cap) →
  `vsr_main.py --database Rioplatense --scenario zero-shot --load-vsr <ckpt>`.

## Pipeline de entrenamiento

`train_orchestrator.sh` (en `scripts/`), corrido en la VM: PHASE_SETUP (env
vsr-factors + Zenodo) → PHASE_DATA (datos) → PHASE_SPLITS (armar_splits, valida
test≈658) → SMOKE → FT03 → FT04 → EVAL (best-effort, 4 checkpoints) → cierre
(snapshot del disco, copia de logs/resultados a `full-run/train/`, apagado de VM).

## Pendientes conocidos

- 5 fuentes de la Fase A cayeron por bot-block/age-gate de YouTube (recuperables con
  cookies u otra IP): `aQlIHv_K0zk`, `ZhPgRWjBWvk`, `scQ7nPWsA8g`, `zyr7wpiIt18`,
  `wB4JpMNFqb4`.

## Scripts (durables, en `vsr/historical/ronda2/scripts/`)

`autograb_train.sh` (auto-grab de GPU + lanzamiento), `train_orchestrator.sh`
(pipeline en la VM), `setup_and_run.sh` + `filtro_musica.py` (Fase A), `autograb_launch.sh`
(Fase A).
