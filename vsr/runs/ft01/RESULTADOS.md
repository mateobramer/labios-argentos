# ft01 — Fine-tuning v1 (baseline, full model)

**Fecha:** 2026-06-25 · **VM:** `labios-vsr-gpu` (GCP, L4) · **Punto de partida:** checkpoint
`vsr-liprtve-si.pth` de Gimeno (ESPnet, Conv3D-ResNet18 + Conformer + CTC/attention, char).

## Resultado titular

Mismo test (200 clips: 100 LE DIJE + 100 ME ACUSARON), mismas settings (beam 10, GPU),
solo cambia el checkpoint:

| | Zero-shot (original) | **Fine-tuned (v1)** | Δ |
|---|---|---|---|
| **%WER** | 80.62 ± 2.17 | **75.17 ± 2.23** | **−5.45** |
| **%CER** | 48.33 ± 1.57 | **45.15 ± 1.74** | **−3.18** |

**Significativo:** los IC no se solapan (ZS [78.4–82.8] vs FT [72.9–77.4]). La mejora es real.

## Config del entrenamiento

- **Estrategia:** fine-tune del **modelo completo** (sin congelar), **sin augmentation**, sin
  regularización (stochastic-depth desactivado por un bug del repo en modo train).
- Optimizer AdamW, **lr 1e-4**, batch 1 + **grad accumulation 8** (efectivo 8), grad clip 5.0.
- Filtro: clips > 400 frames excluidos (memoria; 0.6%).
- Early-stopping (paciencia 5) sobre val_loss.

## Datos

| Split | Clips | Horas |
|---|---|---|
| train | 4818 (≤400f) | **6.72 h** |
| val | 466 | 0.62 h |
| test (curado total) | 658 | 0.83 h |
| **eval de este run** | 200 (subset) | — |

Total curado: 5950 clips / 8.22 h. Crudo antes del preproc: ~13.6 h (se perdió ~40% por
clips sin cara frontal). **Es poco para VSR** (LIP-RTVE = ~13 h; benchmarks inglés 200–440 h).

## Hallazgo

**Overfitting rápido:** val_loss tocó fondo en el **epoch 3** (52.26) y subió después
(53.1 → 54.2 → 55.0) mientras el train loss se desplomaba. Era esperable: 52.5M params sobre
6.7 h. El `best.pth` es el del epoch 3.

Conclusión: con tan pocos datos, el fine-tune full es subóptimo. Igual mejoró ~5.5 pts → el
approach sirve; falta exprimirlo mejor (ver v2).

## Artefactos

- **Modelo:** `gs://labios-argentos-vsr-data/models/ft01_v1/best.pth` (gitignored localmente).
- **Predicciones:** `eval_{zeroshot,finetuned}.inf` (pares `ref#hyp`) · **WER:** `eval_*.wer`.
- **Log de entrenamiento:** `train.log`.

## Próximo: v2

Fine-tune **conservador** para reducir overfitting con los mismos datos:
- Congelar el frontend visual (Conv3D-ResNet18) y/o el encoder.
- Data augmentation (random crop, flip horizontal, time-masking).
- lr más bajo.

Objetivo: superar el 75.17 de WER de la v1.
