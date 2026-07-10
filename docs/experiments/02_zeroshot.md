# 02 — Zero-shots (sin fine-tune argentino)

Evaluar modelos pre-entrenados directamente sobre `test-658`, alimentando nuestros npz (T,96,96, gris,
mean-face 96×96) al VideoTransform del modelo. Sin tocar pesos.

## Resultados (test-658)

| Modelo | Pretrain | Tokenizer | %WER | %CER | Harness |
|---|---|---|---|---|---|
| mpc001 CMU-MOSEAS ES | 16h (CMU-MOSEAS) | char 37 + RNNLM | 71.50 ± 1.70 | 46.88 ± 1.24 | `vsr/mpc001/scripts/zeroshot.py` (env `mvsr`) |
| **ViSpeR** ⭐⭐ | **794h español** | SPM 21k, token `<es>` | **45.22 ± 1.90** | **26.98 ± 1.27** | `~/Desktop/visper/visper_zeroshot.py` (env `visper`) |

## Detalles

**mpc001 (CMU-MOSEAS ES):** misma familia que Gimeno (Conv3D+ResNet18→Conformer→CTC/Att). Único modelo
VSR español liberado de esa línea (WER 44.5 en su propio test CMU-MOSEAS). Sobre rioplatense zero-shot =
71.5, PEOR que ft05 (65) → base domain-matched fine-tuneada > multilingüe genérico zero-shot. Beam 30 + RNNLM.
Preproc idéntico al nuestro (mismo `20words_mean_face.npy`, mismo MD5).

**ViSpeR (TII/Falcon, arXiv:2406.00038):** `visper_vsr_base.pth`, 288M, adim768, SPM unigram 21k con
language tokens (`<es>`=4), espnet1 vendoreado, licencia CC BY-NC. Pre-entrenado con ~794h de español
(207h TEDx + 587h "wild"). **Zero-shot 45.22 WER** = le gana a NUESTRO mejor fine-tune (ft05, 65) por
~20 WER, y a mpc001 (71.5) por ~26. Corre nuestros npz directo (VideoTransform test: CenterCrop88 +
Normalize 0.421/0.165 + beam es). CPU ~6s/clip.

## Conclusión

**El zero-shot de ViSpeR es el mejor punto de partida del proyecto.** Domina la escala de pre-entrenamiento
visual español (794h), no la arquitectura. Reorientó la estrategia: partir de ViSpeR, no del currículum
desde cero. (En condiciones LIMPIAS —self-test del usuario— ViSpeR baja a **23.6 WER / 11.4 CER**; ver [05](05_selftest_limpio.md).)
