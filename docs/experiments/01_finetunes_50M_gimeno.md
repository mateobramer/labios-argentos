# 01 — Fine-tunes del modelo 50M (Gimeno / LIP-RTVE)

**Repo/receta:** `github.com/david-gimeno/evaluating-end2end-spanish-lipreading` (espnet2). Entrenador
propio `vsr/src/fine_tune.py` (reusa transforms/tokenizer/collate de Gimeno + loop propio).
**Base:** `vsr-liprtve-si.pth` (LIP-RTVE speaker-independent, TV española ~13h) — del bundle Zenodo
(record `17443293`, ~8.5GB). Entorno conda `vsr-factors`. Normalización: `Normalise(0,250)` →
`Normalise(0.491,0.166)` → `CenterCrop(88)`. Seed 1234.

**Configs:**
- **v1** = full fine-tuning (sin congelar, sin augment): `--lr 1e-4 --batch 1 --accum 8 --max-frames 400 --paciencia 5`
- **v2** = frontend congelado + augment: `--freeze frontend --augment` (+ los mismos flags)

## Resultados (test-658)

| Modelo | Base | Ronda / config | Train (clips) | Train (h aprox) | %WER | %CER |
|---|---|---|---|---|---|---|
| ft03 | LIP-RTVE | ronda-1, v1 | 4818 | ~5h | 68.93 ± 1.27 | 41.01 ± 0.96 |
| ft04 | LIP-RTVE | ronda-1, v2 | ~4800 | ~5h | 69.73 ± 1.30 | 42.29 ± 0.95 |
| **ft05** ⭐ | LIP-RTVE | ronda-2, v1 | 10934 | ~12-19h | **65.05 ± 1.29** | **38.24 ± 0.97** |
| ft06 | LIP-RTVE | ronda-2, v2 | ~10900 | ~12-19h | 66.37 ± 1.31 | 39.34 ± 0.90 |
| ft05b | LIP-RTVE | same-data, v1 | 8067 | 8.64h (medido) | 70.30 ± 1.30 | 42.08 ± 0.95 |
| ft07 | multiling. remap | same-data, v1 | 8067 | 8.64h | 69.15 ± 1.28 | 41.66 ± 0.91 |

Notas:
- **ronda-1 (ft03/04)** vs **ronda-2 (ft05/06)**: misma receta, más datos.
- **same-data (ft05b/ft07)**: A/B base-vs-base con train IDÉNTICO (8067) y mismo seed. ft07 usa el base
  multilingüe de mpc001 **remapeado espnet1→espnet2** (767/767 tensores). Los números absolutos no igualan
  a ft05 (65) porque el train es menor (8067 vs 10934; el texto de las otras fuentes de ronda-2 se perdió).
- ft05 es el **mejor propio** pero su `.inf` no quedó guardado; ft05b es su gemelo reproducible.

## Conclusiones

1. **Más datos = mejora significativa:** ft05 vs ft03 −3.9 WER (IC no solapan). ft06 vs ft04 −3.4 WER.
2. **v1 (full-FT) ≥ v2 (freeze+aug)**, pero diferencia no significativa.
3. **Base LIP-RTVE ≈ base multilingüe** con datos idénticos (ft07 apenas mejor que ft05b, IC solapan).
   → **los datos importan mucho más que la base.**
4. Fine-tunear (65) supera al zero-shot multilingüe chico (71.5) → adaptar al acento paga… pero todo el
   linaje queda ~20 WER por encima del zero-shot de ViSpeR (45).

## Pendiente / próximo (ft09)
Fine-tune del 50M con datos ampliados (existente 8067 + clips nuevos de Martín) — ver
`docs/HANDOFF_ROIS_FINETUNE.md`. Esperado ~ft05 (65-70), es baseline reproducible para el paper.
