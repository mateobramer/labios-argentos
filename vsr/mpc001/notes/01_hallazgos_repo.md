# mpc001 — Visual Speech Recognition for Multiple Languages — hallazgos

Repo: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
Paper: "Visual Speech Recognition for Multiple Languages" (Ma et al., 2022, arXiv:2202.13084),
sucesor de "End-to-End AVSR with Conformers" (2021). Licencia: solo comparación/benchmark, no comercial.

## Modelos VSR disponibles (model zoo)
| lang | dataset train | métrica | tamaño | link |
|---|---|---|---|---|
| **Spanish** | **CMU-MOSEAS** | **44.5 WER** | 186 MB | GoogleDrive https://bit.ly/34MjWBW (BaiduDrive bit.ly/33rMq3a key m35h) |
| Portuguese | CMU-MOSEAS | 51.4 WER | 186 MB | |
| French | CMU-MOSEAS | 58.6 WER | 186 MB | |
| Mandarin | CMLR | 8.0 CER | 195 MB | |
| English | LRS3 | 32.3 / 19.1 WER | 186 MB | |
| English | LRS2 | 26.1 WER | 186 MB | |
| GRID / TCD-TIMIT / Lombard | varios | 1.2–21.8 WER | 186 MB | |

- **Único modelo VSR en español = CMU-MOSEAS ES (44.5 WER).** mTEDx `es` existe SOLO como labels
  de benchmark (`benchmarks/MultilingualTEDx/labels/es/*.txt`), NO hay pesos liberados para mTEDx.
- El 44.5 es in-domain sobre CMU-MOSEAS. Sobre nuestro test rioplatense va a ser zero-shot
  cross-domain → esperar bastante peor (acento + habla espontánea de YouTube + condiciones).

## Arquitectura
Es la **misma familia** que estamos usando, NO una arquitectura distinta:
- frontend Conv3D + ResNet18 → encoder **Conformer** → decoder híbrido **CTC/Attention**.
- El repo de Gimeno (`evaluating-end2end-spanish-lipreading`) es una adaptación de ESTE codebase
  (Auto-AVSR / ESPnet) a LIP-RTVE. O sea comparten linaje.

Las diferencias reales (no arquitectónicas) son:
1. **Corpus de entrenamiento**: mpc001 ES = CMU-MOSEAS español; nuestra base = LIP-RTVE (TV RTVE).
2. **Tokenizer**: mpc001 = subword unigram5000 (SentencePiece, `pipelines/tokens/unigram5000_units.txt`);
   nuestro = char-level.
3. **LM externo**: mpc001 trae un RNNLM (`lm_weight=0.4`, `ctc_weight=0.1`, `beam_size=30`);
   nuestro decodifica CTC/attention sin LM externo.

## Preprocesamiento — CHEQUEO DE COMPATIBILIDAD: PASA ✅
- `pipelines/detectors/mediapipe/video_process.py` (mpc001) == nuestro
  `preprocessing/src/video_process.py` (adaptación verbatim, mismo código).
- `20words_mean_face.npy`: **MD5 idéntico** en ambos repos (ada7359b793f3406d90fb0fcf2dde069).
- Mismo pipeline: alineación afín a cara media (4 puntos estables ojo-D/ojo-I/nariz/boca),
  crop 96×96, **grises**, 25 fps.
- Normalización en inferencia (`pipelines/data/transforms.py`, VideoTransform):
  `CenterCrop(88)` → `/255` → `Normalize(mean=0.421, std=0.165)`.

**Consecuencia clave:** nuestros npz de test-658 YA están en la convención exacta que espera el
modelo de mpc001. Podemos alimentarlos directo (solo aplicando VideoTransform). NO hace falta
re-descargar videos ni re-preprocesar → zero-shot barato y válido.

## Cómo se corre
- Inferencia: `python infer.py config_filename=configs/CMUMOSEAS_V_ES_WER44.5.ini data_filename=...`
- Config ES (`configs/CMUMOSEAS_V_ES_WER44.5.ini`): input v_fps=30, model v_fps=25 (resamplea),
  model_path=benchmarks/CMUMOSEAS/models/es/..., rnnlm=.../lm_es/model.pth,
  decode beam=30 penalty=0 ctc_weight=0.1 lm_weight=0.4.
- Deps: Python 3.8, torch/torchvision/torchaudio, ffmpeg, RetinaFace o MediaPipe. Basado en ESPnet.

## Plan ejecutado (sin GPU primero)
1. [hecho] Estudio: arquitectura / benchmarks / preproc → este doc.
2. [hecho] Setup: clonar repo en vsr/mpc001/, env de requirements.txt, bajar modelo ES + LM.
3. [hecho] Zero-shot: test-658 npz → VideoTransform → model.infer (beam+LM) → WER/CER
   (resultado en [`docs/experiments/02_zeroshot.md`](../../../docs/experiments/02_zeroshot.md)).
4. Fine-tune del ES sobre nuestros ~12k clips no se hizo: el hallazgo de ViSpeR zero-shot
   (muy por encima de esta familia) hizo que escalar esta línea dejara de ser prioritario.

---

## RESULTADO ZERO-SHOT (2026-07-04) — CMU-MOSEAS ES sobre test-658 rioplatense

**%WER: 71.50 ± 1.70   |   %CER: 46.88 ± 1.24**  (beam=30 + RNNLM, config oficial 44.5)

Setup usado: espnet vendoreado en ~/Desktop/Visual_Speech_Recognition_for_Multiple_Languages
(el de pip es audio-only, sin frontend conv3d). Modelo char-level (char_list 37: alfabeto+áéíóúñü,
SIN dígitos). ctc_type warpctc→builtin. Limpieza de token '<space>'. Normalización idéntica a refs
(lower, sin acentos, ñ ok). Harness: vsr/mpc001/scripts/zeroshot.py. ~12s/clip en CPU.

Comparación (mismo test-658, mismos 2 hablantes held-out f22/f15):
| modelo | base | datos FT | %WER | %CER |
|---|---|---|---|---|
| **CMU-MOSEAS ES (zero-shot)** | CMU-MOSEAS castellano | 0 | **71.50±1.70** | **46.88±1.24** |
| ft03 v1 | LIP-RTVE | ~9k | 68.93±1.27 | 41.01±0.96 |
| ft04 v2 | LIP-RTVE | ~9k | 69.73±1.30 | 42.29±0.95 |
| ft05 v1 | LIP-RTVE | ~12k | 65.05±1.29 | 38.24±0.97 |
| ft06 v2 | LIP-RTVE | ~12k | 66.37±1.31 | 39.34±0.90 |

Significancia (IC95%):
- ES zero-shot vs ft05: IC NO solapan → ft05 significativamente mejor (−6.5 WER, −8.6 CER).
- ES zero-shot vs ft03/ft04: IC solapan → estadísticamente empatado con nuestros fine-tunes más flojos.
- Y eso que el ES tiene ventaja de RNNLM+beam30; nuestros ft eval sin LM externo.

Lectura: el base multilingüe es un VSR fuerte (empata a nuestros ft tempranos zero-shot), pero
nuestros fine-tunes sobre la base domain-matched (LIP-RTVE, TV española) rinden mejor en rioplatense.
El CER peor (46.9 vs 38.2) confirma más errores de carácter por mismatch de dominio/acento.
Mejores clips igual salen muy bien (WER 6-15) → el modelo transfiere, solo falta adaptación.
