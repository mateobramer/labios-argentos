# evaluation

Métricas, baselines y reportes de WER del proyecto (ver `ESTRUCTURA_PROYECTO.md`).

## Baseline zero-shot con el VSR español de Gimeno

Mide cuánto WER da un modelo de lectura de labios entrenado en español **peninsular**
([`david-gimeno/evaluating-end2end-spanish-lipreading`](https://github.com/david-gimeno/evaluating-end2end-spanish-lipreading),
checkpoint **LIP-RTVE speaker-independent**) evaluado **sin fine-tuning** sobre nuestros
clips rioplatenses. Es el punto de partida del proyecto.

El modelo es ESPnet (Conv3D-ResNet18 → Conformer → CTC+attention), `token_type: char`.
Consume ROIs labiales `(T,96,96)` gris a 25 fps en `.npz` y texto `lower+unidecode+ñ`
(idéntico a nuestro `limpiar()`).

### Resultados (149 clips, 2 hablantes) — ver `outputs/RESULTADOS.md`

| Decoding | %WER | %CER |
|---|---|---|
| **sin LM** (titular) | **79.55 ± 2.49** | **49.57 ± 2.09** |
| con LM (peninsular) | 80.21 ± 2.79 | 52.46 ± 2.24 |

Referencia in-domain del modelo: 59.5% WER (LIP-RTVE-si). Degradación ~20 pts al
rioplatense de YouTube, pero CER ~50% ⇒ **lee labios de verdad**. El LM peninsular **no
ayuda** (sesgo léxico ajeno) ⇒ argumento a favor del corrector LLM propio del proyecto.

### Flujo (todo en la VM `labios-vsr-gpu`)

0. **Setup del modelo** (clona repo + baja checkpoints de Zenodo + entorno + parches):
   `bash evaluation/setup_modelo_gimeno.sh`. Los pesos (8.5 GB) no se versionan; se bajan ahí.
1. **ROIs**: `python -m visual_preprocessing.src.preprocesar "<titulo>"` por cada fuente
   del subset → `data/processed/lip_rois/<titulo>/clip_NNNN.npz` (+ `.mp4` para QA).
   (`PREPROC_MAX=N` limita los clips por fuente.)
2. **Export**: `python -m evaluation.src.exportar_para_gimeno --salida ~/data
   --max-por-fuente 80 "<fuente A>" "<fuente B>"` → arma el layout que espera el
   evaluador en `~/data/Rioplatense/{ROIs,transcriptions,splits/zero-shot}` + `mapeo.csv`.
3. **Parches**: `python evaluation/gimeno_patches/aplicar_parches.py ~/evaluating-end2end-spanish-lipreading`
   (registra la base `Rioplatense`; usa mean/std de LIP-RTVE). Ya lo corre el paso 0.
4. **Inferencia** (desde la raíz del repo de Gimeno, env `vsr-factors`):
   ```bash
   CKPT=~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth
   # SIN LM (número titular)
   python vsr_main.py --database Rioplatense --scenario zero-shot \
     --load-vsr $CKPT \
     --output-dir ./spanish-benchmark/rioplatense/liprtve-si_noLM/
   # CON LM (comparación)
   python vsr_main.py --database Rioplatense --scenario zero-shot \
     --load-vsr $CKPT --load-lm ~/zenodo/extracted/Factors_*/LM/lm-liprtve.pth \
     --output-dir ./spanish-benchmark/rioplatense/liprtve-si_LM/
   ```
   Produce `inference/test.inf` (`ref#hyp` por clip) y `test.wer` (WER/CER ± IC).

### Validación (gate)

El Zenodo trae *landmarks*, no los videos de LIP-RTVE, así que reproducir su 59.5% queda
diferido (requeriría bajar el corpus aparte). Gate cualitativo usado: el modelo produce
**español coherente correlacionado con la referencia**, y nuestro crop usa el **mean-face
de Auto-AVSR idéntico** al de sus checkpoints. Detalle en `outputs/RESULTADOS.md`.

### Entorno (gotchas que costaron)

- Python **3.8** ⇒ numpy máx **1.24.4** (1.25+ pide py3.9).
- `espnet` con `--no-build-isolation`; `ctc-segmentation` pide Cython<3, `pyworld` pide
  Cython>=3 (instalar en ese orden).
- `typeguard==2.13.3` (el código de Gimeno usa `check_argument_types`, removido en typeguard 3).
- mediapipe necesita `libGLESv2.so.2`/`libEGL` (apt). WER por binarios C `tasas/`.

### Archivos

- `setup_modelo_gimeno.sh` — setup reproducible del modelo (clone + Zenodo + env + parches).
- `src/exportar_para_gimeno.py` — ROIs `.npz` → layout del evaluador + `splits` + `mapeo.csv`.
- `gimeno_patches/aplicar_parches.py` — parches idempotentes al repo de Gimeno.
- `outputs/` — `RESULTADOS.md`, `liprtve-si_noLM/`, `liprtve-si_LM/` (`test.inf`/`test.wer`), `mapeo.csv`.
- `data/` — datos derivados (gitignored).
