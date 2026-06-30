# ft02 — Fine-tuning v2 (frontend congelado + augmentation)

**Fecha:** 2026-06-25 · **VM:** `labios-vsr-gpu` (GCP, L4) · **Punto de partida:** mismo
checkpoint `vsr-liprtve-si.pth` de Gimeno que la v1.

## Resultado titular

Mismo test que la v1 (200 clips: 100 LE DIJE + 100 ME ACUSARON), mismas settings de eval
(beam 10, GPU). Solo cambia el checkpoint:

| | Zero-shot | v1 (full FT) | **v2 (frozen+aug)** | Δ vs v1 |
|---|---|---|---|---|
| **%WER** | 80.62 ± 2.17 | 75.17 ± 2.23 | **72.24 ± 2.50** | **−2.94** |
| **%CER** | 48.33 ± 1.57 | 45.15 ± 1.74 | **42.92 ± 1.67** | **−2.23** |

**v2 es la mejor versión.** Frente al zero-shot la mejora es **sólida y significativa**
(IC v2 [69.7–74.7] vs ZS [78.5–82.8], no se solapan). Frente a la v1 la dirección es clara
y consistente, pero los IC **se solapan** ([69.7–74.7] vs [72.9–77.4]): con 200 clips no
alcanza para *demostrar* que v2 > v1. Para afirmarlo fuerte: evaluar sobre el test completo (658).

## Config del entrenamiento

- **Estrategia conservadora** para reducir el overfitting que tuvo la v1:
  - **Frontend visual congelado** (Conv3D-ResNet18) → menos params que ajustar.
  - **Data augmentation** en train: `RandomCrop(88)` + `RandomFlip(0.5)` (la v1 usaba CenterCrop fijo).
- Optimizer AdamW, **lr 1e-4**, batch 1 + **grad accumulation 8**, grad clip 5.0.
- Filtro: clips > 400 frames excluidos (memoria). Early-stopping (paciencia 5) sobre val_loss.
- Mismos splits speaker-independent que la v1.

## Hallazgo clave: val_loss ≠ WER

| | mejor val_loss | %WER test |
|---|---|---|
| v1 | **52.26** (ep3) | 75.17 |
| v2 | 53.48 (ep3) | **72.24** |

La v2 tuvo **peor val_loss** (53.48 vs 52.26) pero **mejor WER**. La regularización
(congelar + augment) hace que ajuste un poco peor el loss de val pero **generalice mejor** a
hablantes no vistos. El val_loss subestimó a la v2: el WER en test es la métrica que decide.

La v2 igual overfittea (val_loss tocó fondo en ep3 y subió: 54.4 → 53.6 → 54.8 → 55.0), solo
que su `best.pth` de ep3 generaliza mejor que el de la v1. Se cortó por early-stopping
(se forzó el corte tras ep8, ya sin mejora desde ep3).

## Datos

Mismos que la v1: train 4818 clips (≤400f) / **6.72 h**, val 466, test 200 (subset de 658).
Sigue siendo poco para VSR (LIP-RTVE ~13 h; benchmarks inglés 200–440 h).

## Artefactos

- **Modelo:** `gs://labios-argentos-vsr-data/models/ft02_v2/best.pth` (gitignored localmente).
- **Predicciones:** `eval_finetuned_v2.inf` (pares `ref#hyp`) · **WER:** `eval_finetuned_v2.wer`.
- **Log de entrenamiento:** `train.log`.

## Próximo: v3

La v2 confirma que **regularizar ayuda**. Ideas para exprimir más con los mismos datos:
- Variar la dosis: augment **sin** congelar, o congelar **frontend+encoder** (más agresivo), o lr más bajo.
- Más augmentation (time-masking).
- Evaluar el ganador sobre el **test completo (658)** para un número con IC más ajustado.
- Si el VSR puro se estanca, un **corrector LLM** sobre las hipótesis (post-proceso).

Objetivo: bajar del 72.24 de WER de la v2.
