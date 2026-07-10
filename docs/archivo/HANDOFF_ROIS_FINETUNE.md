# HANDOFF — Extracción de ROIs + Fine-tune del modelo de Gimeno (ft09)

> **Para:** el compañero que continúa el TP + su Claude Code.
> **Objetivo en una línea:** sacar los ROIs de los clips nuevos, y con **todos** los datos
> (viejos + nuevos) **repetir exactamente el fine-tune que veníamos haciendo** (ft05), respetando
> los splits congelados, evaluar sobre el mismo test y comparar.
>
> **Regla de oro:** NO cambies el pipeline ni los splits. La comparación solo vale si es idéntica
> a lo anterior. Este documento tiene los valores exactos para que sea reproducible.

---

## 0. TL;DR (los 3 pasos)

1. **Extraer ROIs** de los 13.193 clips nuevos (`clips_with_audio`) → `.npz` 96×96, con el pipeline
   RetinaFace de auto_avsr/mpc001. GPU (L4).
2. **Armar splits**: `test` y `val` quedan CONGELADOS (idénticos); a `train` se le suman TODOS los
   clips nuevos que hayan salido con ROI (los nuevos van 100% a train, sin excepción).
3. **Fine-tune ft09** = receta v1 full-FT de ft05 (`--lr 1e-4 --batch 1 --accum 8 --max-frames 400
   --paciencia 5 --seed 1234`), evaluar sobre `test-658`, comparar contra **ft05b (WER 70.30)**.

> El **preprocesamiento anterior a los ROIs** (descarga → Whisper → segmentación en clips → limpieza
> → QA) NO hace falta en el flujo normal (Martín ya generó clips + transcripciones), pero está
> **documentado completo en el Apéndice A** por si hay que reprocesar o sumar fuentes.

---

## 1. Contexto (qué se hizo antes)

- **Modelo base:** repo de **David Gimeno** `evaluating-end2end-spanish-lipreading`
  (https://github.com/david-gimeno/evaluating-end2end-spanish-lipreading), checkpoint
  **`vsr-liprtve-si.pth`** (LIP-RTVE speaker-independent, ~102M params). Arquitectura:
  Conv3D+ResNet18 → Conformer (12 capas) → decoder híbrido CTC/Attention. Offline/bidireccional.
- El checkpoint sale de un **bundle de Zenodo** (~8.5 GB, carpeta `Factors_*`), NO del GitHub.
- **Modelos entrenados (todos sobre test-658, speaker-independent, split congelado):**

  | Modelo | Train (clips) | Receta | %WER | %CER |
  |---|---|---|---|---|
  | ft03 | 4818 | v1 full-FT | 68.93 | 41.01 |
  | ft04 | ~4800 | v2 freeze+aug | 69.73 | 42.29 |
  | **ft05** ⭐ | 10934 | v1 full-FT | **65.05** | 38.24 |
  | ft06 | ~10900 | v2 freeze+aug | 66.37 | 39.34 |
  | **ft05b** | 8067 | v1 full-FT | **70.30** | 42.08 |
  | ft07 | 8067 | v1 (base multiling.) | 69.15 | 41.66 |

- **Baseline de comparación = ft05b (70.30 / 42.08).** ¿Por qué no ft05 (65.05)? Porque el train
  exacto de ft05 (10934) usaba transcripciones que **se perdieron** → ft05 NO es reproducible.
  ft05b es su gemelo riguroso (train 8067 recuperable, mismo seed y receta). El **ft09** que vas a
  entrenar = 8067 viejos + los nuevos, así que el delta contra ft05b es el **efecto puro de sumar
  datos nuevos**.

---

## 2. Datos — dónde está todo

**Bucket principal:** `gs://labios-argentos-vsr-clean-v1/` (proyecto `labios-argentos-499900`;
tenés lectura como `fgutman`). Estructura relevante:

| Qué | Path | Detalle |
|---|---|---|
| **Viejos (ya son ROIs)** | `argentina/existing/rois_npz/` | **12.112** `.npz` 96×96. Splits ya asignados: train 8067 / val 466 / test 658. |
| **Nuevos (hay que sacar ROIs)** | `argentina/new_discovery/clips_with_audio/` | **13.193** `.mp4` de **cuadro completo + audio** (NO son ROIs todavía). |
| ROIs nuevos ya hechos | `argentina/new_discovery/rois_npz/` | solo **2.248** (17%). El resto falta. |
| **Transcripciones nuevas** | `argentina/new_discovery/transcripts/` | ⚠️ Martín va a subir las **CORREGIDAS**. Usá esas, NO el ASR crudo. |
| Manifest maestro | `manifests/final_release_manifest.csv` | columnas: `dataset_group, source_id, clip_id, split, spk, titulo, ..., npz_path, selected_training_text, alignment_confidence, usable_for_training, needs_review, ...` |

**Copia de los clips nuevos en nuestro bucket (staging listo, por si el otro bucket da problemas de
permisos desde la VM):** `gs://labios-argentos-vsr-dataset/new_discovery/clips_with_audio/` (13.193).
Estructura: `<video_id>/clip_XXXX.mp4`.

### 2.1 Buckets y autenticación — LEER ANTES DE EMPEZAR

Hay **dos buckets vivos** (ignorá cualquier referencia a un tercero, `gs://labios-argentos-vsr-data`,
que está muerto/403 en scripts viejos):

| Bucket | Proyecto | Acceso desde la VM | Qué tiene |
|---|---|---|---|
| **`gs://labios-argentos-vsr-dataset`** (NUESTRO) | `visual-speech-recognition-nlp` | **lectura Y escritura** (service account propia) | splits congelados, ROIs viejos, modelos, config/scripts, **staging de los clips nuevos** |
| `gs://labios-argentos-vsr-clean-v1` (de Martín) | `labios-argentos-499900` | **solo lectura** como usuario `fgutman`; la **service account de una VM NO puede leerlo** (otro proyecto, sin IAM) | clips nuevos originales + manifests + transcripciones corregidas |

> **Regla práctica:** hacé **TODO desde `gs://labios-argentos-vsr-dataset`**. Por eso los clips nuevos
> ya están copiados ahí (staging). Si necesitás algo de clean-v1 (p. ej. las transcripciones
> corregidas de Martín), bajalo **con tu cuenta `fgutman`** a tu máquina o a nuestro bucket, y desde
> ahí lo lee la VM. No intentes que la VM lea clean-v1 directo: falla por IAM (ya nos pasó).

### 2.2 Dónde está cada insumo (todo en `gs://labios-argentos-vsr-dataset`)

| Insumo | Ubicación | Nota |
|---|---|---|
| **Splits congelados** | `splits/{splits.csv,train.csv,val.csv,test.csv}` | **YA ARMADOS** (los mismos de ft05b): train **8067** / val **466** / test **658**. Columnas: `split,spk,titulo,clip,n_frames,texto,npz`. **Usar tal cual** (ver §4). |
| **ROIs viejos** (12.112) | `lip_rois/<titulo>/clip_NNNN.npz` | los de Federico, ya croppeados; se bajan a `rois_root` de la VM. |
| Clips nuevos (staging) | `new_discovery/clips_with_audio/<video_id>/clip_XXXX.mp4` | entrada de la extracción de ROIs (Parte 1). |
| Scripts/eval de referencia | `config/` | `train_ab.sh` (la corrida que produjo ft05b), `ft05b_test.wer`, etc. |

### 2.3 Scripts que viven en el REPO (no en un bucket)

El harness de eval y el setup están **en el repo `labios-argentos`** bajo `evaluation/`
(`setup_modelo_gimeno.sh`, `src/exportar_para_gimeno.py`, `gimeno_patches/aplicar_parches.py`) y el
trainer en `vsr_models/src/fine_tune.py`. **Si al clonar/pullear no aparece `evaluation/`**, restaurá
esos archivos desde git history (estuvieron en el commit `6be9bd55`):

```bash
git show 6be9bd55:evaluation/setup_modelo_gimeno.sh          > evaluation/setup_modelo_gimeno.sh
git show 6be9bd55:evaluation/src/exportar_para_gimeno.py     > evaluation/src/exportar_para_gimeno.py
git show 6be9bd55:evaluation/gimeno_patches/aplicar_parches.py > evaluation/gimeno_patches/aplicar_parches.py
```

> **Sin fuga de datos (verificado):** los hablantes de test/val son `f02,f05,f15,f22,f37`; los clips
> nuevos son de canales (El Método Rebord, Cenital, TN, etc.), identidades distintas → los nuevos
> pueden ir todos a `train` sin contaminar el test.

---

## 3. PARTE 1 — Extracción de ROIs

### 3.1 Pipeline (idéntico a auto_avsr / mpc001, detector RetinaFace)

Código a usar — en la VM cloná el repo mpc001 en `$HOME` (en la máquina de Fede está en
`~/Desktop/...`, pero en la VM usá `~/`):
```bash
git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages ~/vsr_mpc001
```
El pipeline vive en `~/vsr_mpc001/pipelines/detectors/retinaface/`:

- `detector.py` → `LandmarksDetector`: `RetinaFacePredictor(model='resnet50', threshold=0.8)` +
  `FANPredictor` (68 landmarks 2D). Por frame elige la cara **más grande**.
- `video_process.py` → `VideoProcess`: warp affine a **mean-face**, recorte de boca **96×96**,
  `RGB2GRAY`, `start_idx=48 stop_idx=68`. Interpola landmarks faltantes; si ningún frame tiene cara
  devuelve `None` (clip descartado).
- El **`20words_mean_face.npy` viene bundleado** en ese mismo dir (`VideoProcess` lo carga solo por
  ruta relativa) — no hay que bajarlo aparte.
- **Salida esperada por clip (formato EXACTO que espera el fine-tune):**
  `np.savez_compressed(out, rois=seq)` donde `seq` es `(T, 96, 96)` **uint8** (clave del npz = `"rois"`).

### 3.2 Entorno (GPU recomendado — L4)

```bash
# torch/torchvision con CUDA ya vienen en la imagen labios-img-visper (proyecto propio).
pip install av opencv-python scikit-image                      # av = backend de torchvision.io.read_video
pip install git+https://github.com/hhj1897/face_detection.git  # ibug.face_detection (RetinaFace)
pip install git+https://github.com/hhj1897/face_alignment.git  # ibug.face_alignment (FAN)
```

> El detector corre en GPU (por eso conviene L4). Es el paso lento/caro; el fine-tune después es liviano.

### 3.3 Driver de extracción

Primero bajá los clips nuevos desde NUESTRO bucket (la VM puede leerlo; clean-v1 no — ver §2.1):
```bash
gcloud storage cp -r gs://labios-argentos-vsr-dataset/new_discovery/clips_with_audio ~/clips_new
```
Driver (`~/clips_new/<video_id>/clip_XXXX.mp4` → `~/rois_new/<video_id>/clip_XXXX.npz`):
```python
import os, glob, numpy as np, torchvision, sys
sys.path.insert(0, os.path.expanduser("~/vsr_mpc001"))     # <-- donde clonaste el repo mpc001
from pipelines.detectors.retinaface.detector import LandmarksDetector
from pipelines.detectors.retinaface.video_process import VideoProcess

det = LandmarksDetector(device="cuda:0")
vp  = VideoProcess()                      # crop 96x96 gris, mean-face

clips = sorted(glob.glob(os.path.expanduser("~/clips_new/**/*.mp4"), recursive=True))
# PILOT: arrancá con clips = clips[:300] para medir yield antes del run completo (ver §3.4)
ok = fail = 0
for fn in clips:
    out = fn.replace("/clips_new/", "/rois_new/").replace(".mp4", ".npz")
    if os.path.exists(out):               # resumible
        ok += 1; continue
    try:
        lms = det(fn)                                       # lista, None donde no hay cara
        frames = torchvision.io.read_video(fn, pts_unit="sec")[0].numpy()  # (T,H,W,3) RGB
        seq = vp(frames, lms)                               # (T,96,96) uint8 o None
        if seq is None or len(seq) < 12:
            fail += 1; continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, rois=seq.astype("uint8"))
        ok += 1
    except Exception as e:
        fail += 1
        print(f"[fail] {fn}: {e}", flush=True)
print(f"YIELD: {ok}/{ok+fail} = {100*ok/max(ok+fail,1):.1f}%")
```

### 3.4 ⚠️ OJO con el YIELD (dato importante)

Martín solo sacó **2.248 de 13.193 = 17% de yield**. No sabemos si fue una corrida incompleta o si
los clips son duros (tomas abiertas de noticiero, perfiles, varias caras). **Hacé un PILOT de ~300
clips primero y medí el yield antes de gastar el run entero:**

- Si yield ≥ ~40-50% → seguí con los 13.193 completos.
- Si yield ~17% → probá bajar el `threshold` del `RetinaFacePredictor` (0.8 → 0.5/0.6, sube recall) y
  re-corré el pilot. Si igual queda bajo, los clips son difíciles y el techo de "dato nuevo" es bajo
  → avisá antes de seguir.

**Estimado de tiempo (L4):** setup entorno ~15-25 min · pilot 300 ~5-10 min · run completo **~3-6 h**
(depende del rate real que mida el pilot) · upload+teardown ~15 min. **Costo ~US$4-6.**

### 3.5 Salida
Subí los `.npz` nuevos a NUESTRO bucket (a clean-v1 NO tenés escritura):
```bash
gcloud storage cp -r ~/rois_new/* gs://labios-argentos-vsr-dataset/new_discovery/rois_npz/
```
Guardá el **reporte de yield** (cuántos de 13.193 salieron) — sirve para saber cuánto dato nuevo real
entró al train.

---

## 4. PARTE 2 — Armado de splits

> **CLAVE: los splits YA ESTÁN ARMADOS y son los mismos de ft05b.** NO los rearmes desde cero — la
> comparación solo vale si el test/val es idéntico. NO existe ningún `armar_splits.py` que necesites:
> los CSV ya están en el bucket.

**Bajá los splits pre-armados** (de `gs://labios-argentos-vsr-dataset/splits/`):
```bash
gcloud storage cp -r gs://labios-argentos-vsr-dataset/splits/ vsr_models/splits/
# quedan: splits.csv (maestro), train.csv (8067), val.csv (466), test.csv (658)
```

Formato (columnas): `split,spk,titulo,clip,n_frames,texto,npz`. El fine-tune usa `titulo`, `clip`,
`texto`, `n_frames`; arma la ruta como `rois_root/<titulo>/<clip>.npz` (**`titulo` = carpeta**,
**`clip` = nombre sin extensión**). La columna `npz` es cosmética (el trainer la ignora), pero conviene
completarla. `spk` **no lo usa el trainer** (solo importa en el eval, donde el test ya viene fijo).

Qué hacer:
1. **`test.csv` y `val.csv`: NO SE TOCAN.** Ya son los 658 / 466 congelados de ft05b.
2. **`train.csv`**: partí del `train.csv` bajado (8067 viejos) y **agregale una fila por cada clip
   nuevo que haya salido con ROI**, con:
   - `split` = `train`
   - `spk` = un código por fuente nueva (p. ej. el `<video_id>` o el canal; da igual el valor, el
     trainer no lo lee — solo tiene que estar y no colisionar con los `f0x` de test).
   - `titulo` = `<video_id>` (la carpeta del npz nuevo), `clip` = `clip_XXXX`
   - `n_frames` = `np.load(npz)["rois"].shape[0]`
   - `texto` = transcripción **CORREGIDA de Martín**, normalizada (minúsculas, sin puntuación, ñ ok)
   - `npz` = ruta relativa del npz nuevo (cosmética)
3. Los npz nuevos van a `train` en su totalidad (verificado: sus hablantes = canales, distintos de los
   `f02,f05,f15,f22,f37` de val/test → sin leakage).

Chequeo de sanidad antes de entrenar: `test == 658`, `val == 466`, `train > 8067` (si no aumentó, algo
falló en la extracción o en el armado). Y verificá que **todos** los npz referenciados en `train.csv`
existan en `rois_root` (el trainer aborta si falta alguno).

---

## 5. PARTE 3 — Fine-tune ft09 (receta EXACTA de ft05 = v1 full-FT)

**Setup del entorno (un solo comando, self-contained):**
```bash
bash evaluation/setup_modelo_gimeno.sh
```
Ese script deja TODO listo (verificado, está en el repo): clona el repo de Gimeno, baja los
checkpoints de **Zenodo record `17443293`** (~8.5 GB, con verificación md5 `c8adb97d…`), crea el env
conda **`vsr-factors`** (python 3.8, `torch==2.4.1+cu121`, `espnet`, etc.) y **aplica los parches**
(`gimeno_patches/aplicar_parches.py`) que registran la base "Rioplatense" y habilitan `--load-vsr`.
Checkpoint base resultante = `~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth`.

**Armá `rois_root` con viejos + nuevos juntos** (el trainer los espera en un solo árbol
`<titulo>/<clip>.npz`):
```bash
ROIS=~/labios-argentos/data/processed/lip_rois
gcloud storage cp -r gs://labios-argentos-vsr-dataset/lip_rois/*            $ROIS/   # 12.112 viejos
gcloud storage cp -r gs://labios-argentos-vsr-dataset/new_discovery/rois_npz/* $ROIS/  # nuevos (los que sacaste)
```

**SMOKE primero** (valida 1 batch, no quema GPU si la config está mal):

```bash
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config  ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr    "$(ls ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth | head -1)" \
  --rois-root   ~/labios-argentos/data/processed/lip_rois \
  --out         vsr_models/runs/ft09 \
  --smoke
```

**Entrenamiento ft09 (v1 full-FT — SIN `--freeze`, SIN `--augment`):**

```bash
python -m vsr_models.src.fine_tune \
  --gimeno-repo ~/evaluating-end2end-spanish-lipreading \
  --vsr-config  ~/evaluating-end2end-spanish-lipreading/configs/VSR/vsr_conv3dresnet18_conformer_ctc+transformer.yaml \
  --load-vsr    "$(ls ~/zenodo/extracted/Factors_*/VSR/vsr-liprtve-si.pth | head -1)" \
  --rois-root   ~/labios-argentos/data/processed/lip_rois \
  --out         vsr_models/runs/ft09 \
  --lr 1e-4 --batch 1 --accum 8 --max-frames 400 --paciencia 5 --seed 1234
```

Detalles que YA están en `vsr_models/src/fine_tune.py` (no tocar):
- **Corré desde la raíz del repo** (`cd ~/labios-argentos`). El trainer lee los splits de
  `vsr_models/splits/` **hardcodeado** (no hay flag `--splits-dir`), así que ahí tienen que estar los
  `{train,val,test}.csv` que bajaste en §4.
- **Usá el root único `~/labios-argentos/data/processed/lip_rois`** para los ROIs (viejos + nuevos):
  es el mismo que usó `train_ab.sh`, y el exportador de eval (§6) lee de esa misma ruta, así que las
  fuentes de test ya quedan disponibles para el eval sin copiar nada.
- **Normalización:** `Normalise(0, 250)` → `Normalise(0.491, 0.166)` (mean/std rioplatense) → `CenterCrop(88)`.
- **`rois_root`** debe contener **viejos + nuevos** juntos, con estructura `<titulo>/<clip>.npz`.
- `epochs` default 30, **early stopping** por `val_loss` (paciencia 5), guarda `best.pth`.
- Batch efectivo = `batch 1 × accum 8 = 8`. `--max-frames 400` saltea clips larguísimos (memoria del
  Conformer es O(T²)).
- `--seed 1234` (mismo que ft05b → comparación reproducible).

> **NO uses v2** (`--freeze frontend --augment`): eso es ft04/ft06 y da peor. La receta a replicar es
> **v1 full-FT**.

---

## 6. PARTE 4 — Eval sobre test-658 + comparación

Evaluá con **EXACTAMENTE el mismo harness que produjo el 70.30 de ft05b** — si no, la comparación no
vale. Son los comandos tal cual de `config/train_ab.sh` (la corrida de ft05b/ft07). Requiere el repo
de Gimeno **ya parcheado** (lo dejó así `setup_modelo_gimeno.sh` en §5) + los scripts `evaluation/`.

**Paso 1 — exportar el test-658 al formato del evaluador de Gimeno** (las 2 fuentes de test; sus ROIs
tienen que estar en `data/processed/lip_rois/`, o sea entre los 12.112 viejos que bajaste):
```bash
cd ~/labios-argentos
python -m evaluation.src.exportar_para_gimeno --salida ~/data --max-por-fuente 9999 \
  "LE DIJE QUE SOY ARGENTINO - Story Time - CAP 91" \
  "ME ACUSARON DE BRUJA Y ME TUVE QUE IR DEL PUEBLO -"
```

**Paso 2 — correr el evaluador con los pesos de ft09** (el `--load-vsr` sobrescribe el checkpoint; el
`--scenario zero-shot` es solo el nombre del split del benchmark, NO significa que no esté fine-tuneado):
```bash
cd ~/evaluating-end2end-spanish-lipreading
python vsr_main.py --database Rioplatense --scenario zero-shot \
  --load-vsr ~/labios-argentos/vsr_models/runs/ft09/best.pth \
  --output-dir spanish-benchmark/rioplatense/ft09/
# WER/CER en: spanish-benchmark/rioplatense/ft09/inference/test.wer
```

> `vsr_main.py` con `--database Rioplatense` y `--load-vsr` **solo existe tras aplicar los parches**
> (`aplicar_parches.py`: registra "Rioplatense" en `MyDataset.py` con `delimiter=5` + mean/std
> `0.491,0.166`). El repo stock NO los tiene. Si `setup_modelo_gimeno.sh` corrió, ya están aplicados;
> si no, corré `python evaluation/gimeno_patches/aplicar_parches.py ~/evaluating-end2end-spanish-lipreading`.
> La métrica (`norm()` + IC 95% bootstrap) es la misma que veníamos usando.

**Comparar y registrar en `docs/RESULTS.md`:**

| Comparación | Referencia | Qué mide |
|---|---|---|
| ft09 vs **ft05b (70.30 / 42.08)** | mismo recipe, +datos nuevos | **efecto puro de los datos nuevos** (delta limpio) |
| ft09 vs ft05 (65.05 / 38.24) | mejor propio (train mayor, no reproducible) | referencia aspiracional |

---

## 7. Infra / costo (importante)

- Todo en GCP, GPU **L4** (`g2-standard-8` + `nvidia-l4`). Cuota GPU L4 = 1.
- **Borrá la VM y el disco al terminar** (no dejes nada prendido). Monitoreá el costo.
- Extracción de ROIs ~US$4-6; fine-tune ~US$3-8 según epochs. Total esperado < US$15.

---

## 8. Expectativa honesta (para no frustrarse)

- Sumar los datos nuevos (2.248 seguros; más si sube el yield) sobre los 8067 → **mejora modesta
  esperada: ~1-3 WER sobre ft05b (70.30)**, o sea banda **~67-69 WER**. El texto nuevo es de canales
  informales y puede ser ruidoso.
- Esto **NO** compite con el hallazgo del proyecto (ViSpeR zero-shot = **45.22 WER**, sin fine-tune).
  El valor de este ft09 es: (a) **baseline limpio y reproducible** para el paper (curva de datos
  8067 → 8067+nuevos sobre el mismo test), y (b) cerrar bien el experimento del base propio.
- Si el yield de ROIs es alto (muchos clips nuevos), la mejora puede ser mayor — por eso importa el
  pilot y no cortar el yield antes de tiempo.

---

## 9. Checklist para el Claude del compañero

**Prerequisito duro (bloquea el train, NO la extracción de ROIs):** las **transcripciones corregidas
de Martín** tienen que estar subidas antes de armar `train.csv`. La extracción de ROIs (Parte 1) es
independiente del texto y se puede arrancar ya.

- [ ] Verificar que el repo tenga `evaluation/` y `vsr_models/src/fine_tune.py`; si falta `evaluation/`, restaurarlo desde git (§2.3).
- [ ] Clonar repo mpc001 en `~/vsr_mpc001` + instalar entorno RetinaFace (`ibug`) en la L4 (§3.1-3.2).
- [ ] Bajar los clips nuevos de `gs://labios-argentos-vsr-dataset/new_discovery/clips_with_audio`.
- [ ] PILOT 300 clips → medir yield. Bajar `threshold` a 0.5/0.6 si <40% (§3.4).
- [ ] Extraer ROIs de los 13.193 → `.npz` 96×96 (clave `rois`, uint8). Subir a nuestro bucket + guardar reporte de yield.
- [ ] `setup_modelo_gimeno.sh` (env `vsr-factors` + Zenodo + repo Gimeno parcheado).
- [ ] Bajar splits pre-armados de `gs://labios-argentos-vsr-dataset/splits/` (NO rearmar test/val).
- [ ] Confirmar transcripciones CORREGIDAS de Martín → agregar las filas nuevas SOLO a `train.csv`.
- [ ] Armar `rois_root` con viejos (`lip_rois/`) + nuevos juntos. Chequeo: test==658, val==466, train>8067.
- [ ] SMOKE del fine-tune. Si OK → entrenar ft09 (v1 full-FT, flags exactos de §5).
- [ ] Eval con el harness de §6 (exportar + `vsr_main.py` parcheado). Comparar vs ft05b (70.30). Registrar en `docs/RESULTS.md`.
- [ ] Teardown VM + disco. Verificar costo.

---

## Apéndice A — Preprocesamiento upstream completo (por si hay que reprocesar o sumar fuentes)

> **¿Cuándo necesitás esto?** En el flujo normal **NO** — Martín ya produjo los `clips_with_audio`
> (segmentados) y las transcripciones, así que tu compañero arranca directo en la **Parte 1 (ROIs)**.
> Esta sección documenta el pipeline **anterior a los ROIs** por si hace falta **regenerar clips desde
> cero, sumar fuentes nuevas, o re-transcribir**. Todos los scripts están en este repo (`labios-argentos`)
> y corren desde la raíz.

### A.0 Flujo completo (de un video de YouTube a un `.npz` entrenable)

```
URL YouTube
  │  descargar_procesar.py
  ├─(1) yt-dlp  ────────────────►  data/videos/<titulo>.mp4
  ├─(2) Whisper (word_timestamps) ►  data/corpus/<titulo>/transcripcion.json + corpus.txt
  ├─(3) cortar en pausas reales  ►  data/clips/<titulo>/clip_NNNN.mp4 + clip_NNNN.txt   ← "clips_with_audio"
  │
  │  filtro_musica.py            ►  borra clips de música/alucinación
  │
  │  preprocessing/src/preprocesar.py
  ├─(4) MediaPipe landmarks → warp mean-face → 96×96 gris 25fps
  │                              ►  data/processed/lip_rois/<titulo>/clip_NNNN.npz  ← ENTRADA DEL MODELO
  │
  │  cleaning/visual_quality/  (QA, opcional pero recomendado)
  ├─ auditar_alineacion.py       ►  caza drift clip↔texto y errores de Whisper
  └─ detectar_clips_malos.py     ►  marca keep/review/drop (negro/congelado/boca inactiva) → dataset/ curado
```

### A.1 Descarga + transcripción + segmentación — `descargar_procesar.py`

Un solo script hace las 3 primeras etapas. Dependencias: `yt-dlp`, `ffmpeg` (en PATH),
`openai-whisper`, `unidecode`. Modelo Whisper vía env `WHISPER_MODEL` (usamos **`large-v2`** para el
dataset bueno; `turbo` es más rápido y algo peor).

- **(1) Descarga** (`bajar_video`): `yt-dlp` con merge a `mp4` (`bv*+ba/merge`).
- **(2) Transcripción** (`transcribir`): `whisper.transcribe(video, language="es",
  word_timestamps=True, fp16=False)`. **`word_timestamps=True` es CLAVE**: sin tiempos por palabra,
  Whisper solo da tiempos por segmento y el clip queda **corrido respecto de su texto** (deriva
  clip↔texto). Con tiempos por palabra, el rango del clip y su texto salen de **los mismos** timestamps.
- **(3) Segmentación** (`agrupar_palabras` + `cortar_clips`): un clip **se cierra** cuando ya duró
  `DUR_MIN` y aparece una pausa real (silencio entre palabras ≥ `GAP_CORTE`); si no hay pausa, se
  fuerza el corte en `DUR_MAX`. Parámetros EXACTOS:

  | Param | Valor | Qué es |
  |---|---|---|
  | `DUR_MIN` | **3.0 s** | duración mínima antes de permitir corte |
  | `DUR_MAX` | **10.0 s** | corte forzado |
  | `GAP_CORTE` | **0.40 s** | silencio mínimo entre palabras para cortar |
  | `PAD` | **0.08 s** | margen para no comerse el primer/último fonema |

- **(Texto) Normalización** (`limpiar`): `lower()` → quita puntuación (`re.sub(r"[^\w\s]", "")`) →
  preserva la **ñ** (truco ENIE) y saca el resto de acentos con `unidecode`. Cada clip guarda su `.txt`.

Uso:
```bash
WHISPER_MODEL=large-v2 python descargar_procesar.py <URL_o_lista>   # ver el bloque __main__ del script
```

### A.2 Filtro de música/alucinación — `new-data-fine-tuning/scripts/filtro_musica.py`

Whisper alucina sobre música/canto (frases repetidas). Borra clips cuyo texto tenga **baja variedad
léxica** (`uniq < 0.35`) **o** un **3-grama repetido ≥ 3 veces**. Loguea lo descartado.
```bash
python new-data-fine-tuning/scripts/filtro_musica.py "<titulo>"
```

### A.3 Preprocesamiento visual (ROIs) — `preprocessing/src/preprocesar.py`

**Esta es la variante MediaPipe** del mismo paso que la Parte 1 (que usa RetinaFace). Ambas producen
el **mismo formato** (96×96 gris, warp a mean-face, `.npz` clave `rois` uint8) porque las dos alinean
al mismo `20words_mean_face.npy`. Diferencias: MediaPipe (478 landmarks → 4 puntos estables) corre en
CPU y es la que generó los ROIs **viejos** de Federico; RetinaFace suele dar mejor recall/yield en caras
no frontales (por eso la Parte 1 la prefiere para los clips nuevos). **Cualquiera de las dos sirve y son
intercambiables** — lo importante es no mezclar resoluciones ni fps.

- **Remuestreo a 25 fps OBLIGATORIO** (`FPS_SALIDA=25`, `remuestrear_a_25fps`): el modelo espera 25 fps.
  Si sumás fuentes a otro fps, este paso las normaliza.
- Descarta clips sin cara frontal estable. Reanudable/idempotente (saltea los que ya tienen `.npz`).
- Modelo de landmarks: `preprocessing/models/face_landmarker.task`.
```bash
python -m preprocessing.src.preprocesar --jobs 7          # todas las fuentes, 7 procesos
python -m preprocessing.src.preprocesar "<titulo>"        # una fuente
```

### A.4 Control de calidad (recomendado antes de entrenar) — `cleaning/visual_quality/`

- **`auditar_alineacion.py`** — re-transcribe el audio de cada clip con Whisper y lo compara contra su
  propio `.txt` y los vecinos. Detecta: (a) **drift** (el audio matchea mejor el texto del vecino) y
  (b) **texto dudoso** (WER alto contra su propio texto). Salida:
  `data/metadata/auditoria_alineacion_manifest.csv`.
- **`detectar_clips_malos.py`** — QA a nivel **píxel** sobre los ROIs (no usa MediaPipe): marca
  **negro/oscuro**, **congelado** y **boca inactiva** → estados `keep/review/drop`. Con `--materializar`
  copia solo los `keep` a `dataset/`. Salida: `data/metadata/auditoria_clips_manifest.csv`.
- **`whisper_model_comparison.py`** — compara transcripciones entre modelos de Whisper para decidir cuál
  usar / limpiar.

### A.5 Estructura de salida (la que consume el fine-tune)

```
data/
├── videos/<titulo>.mp4
├── corpus/<titulo>/{transcripcion.json, corpus.txt}
├── clips/<titulo>/clip_NNNN.{mp4,txt}                      # = "clips_with_audio"
├── processed/lip_rois/<titulo>/clip_NNNN.{mp4,npz,txt}     # .npz = (T,96,96) uint8, clave "rois"
└── metadata/*.csv                                          # manifests de preproc + QA
```

- El fine-tune (`fine_tune.py`) espera `rois_root/<titulo>/<clip>.npz`. `titulo` = carpeta, `clip` =
  nombre sin extensión.
- **`n_frames`** (para los splits y `--max-frames`) = `np.load(npz)["rois"].shape[0]`.

### A.6 Dependencias del preproc (además del entorno de ROIs de la Parte 1)

`yt-dlp`, `ffmpeg` (binario en PATH), `openai-whisper`, `unidecode`, `opencv-python`, `mediapipe`
(solo si usás la variante MediaPipe de A.3), `numpy`.

