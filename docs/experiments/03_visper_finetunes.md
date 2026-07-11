# 03 — Fine-tunes de ViSpeR sobre argentino

Partiendo de `visper_vsr_base.pth` (794h español, ver [02](02_zeroshot.md)), fine-tunear sobre ~19h
rioplatenses (8067 clips). Mantiene el tokenizer SPM de ViSpeR. Entrenador
`~/Desktop/visper/fine_tune_visper.py`, en GCP L4 (imagen `labios-img-visper`).

## Resultados (test-658)

| Enfoque | Config | Train | %WER | %CER | vs zero-shot (45.22) |
|---|---|---|---|---|---|
| ViSpeR zero-shot | — | 0 | 45.22 ± 1.90 | 26.98 ± 1.27 | — (referencia) |
| ViSpeR **full-FT** | batch1+accum8, lr1e-4 | 8067 | 61.51 ± 1.93 | 38.59 ± 1.37 | **❌ −16 WER (destroza)** |
| ViSpeR **LoRA+aug** | freeze + LoRA(r16, 0.8% params) + augment + early-stop | 8067 | 45.97 ± 2.02 | 26.00 ± 1.25 | ≈ **empata** (IC solapan) |

## Detalles

**full-FT (2026-07-06):** val_loss subió monótono desde época 1 (39.05→41.16) = **overfit clásico**.
El early-stopping guardó época 1 e igual quedó 16 WER por encima del zero-shot. **Con ~19h no se puede
adaptar un modelo de 794h sin degradarlo** (catastrophic forgetting). Modelo `modelos/ft_visper_ar_best.pth`.

**LoRA+aug (2026-07-06):** freeze base + LoRA r16/alpha32 en las Linears de atención (linear_q/k/v/out,
0.8% params entrenables) + data augment (RandomCrop88 + AdaptiveTimeMask) + early-stop. Resultado:
**45.97 / 26.00 = empata el zero-shot** (WER +0.75, CER −0.98, dentro del ruido; IC solapan). La
regularización correcta **evita la degradación** del full-FT, pero **19h no agregan señal** sobre lo que
ya capturó el pre-entreno de 794h. Modelo `modelos/ft_visper_lora_best.pth`.

## Conclusión

**Con ~19h argentinas no se le gana al zero-shot de ViSpeR.** full-FT overfitea; LoRA+aug empata.
Para superarlo harían falta **muchas más horas** rioplatenses (o el mismo LoRA con +datos). El dataset
argentino NO fue en vano: es el **benchmark** (test-658) que reveló todo. Para producción conviene el
**zero-shot directo** (más simple, empata al LoRA).
