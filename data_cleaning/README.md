# data_cleaning

Auditoria y limpieza del dataset crudo.

Este modulo sirve para revisar si los pares `clip_NNNN.mp4` / `clip_NNNN.txt`
son buenos datos antes de pasarlos al preprocesamiento visual o al entrenamiento.

## Que va aca

- inventarios de clips, textos y fuentes;
- chequeos de pares faltantes entre `.mp4` y `.txt`;
- duracion de clips y textos sospechosos;
- revision de cortes raros, por ejemplo clips que empiezan o terminan en mitad de una palabra;
- manifests de calidad con estados como `keep`, `review` y `drop`;
- scripts o notebooks de limpieza cuando tengan codigo real y resultados reproducibles.

## Que no va aca

- deteccion de rostro, landmarks o recorte de labios: eso va en `visual_preprocessing/`;
- entrenamiento de modelos VSR: eso va en `vsr_models/` si se agrega al repo;
- logica de LLM/correccion en tiempo real: eso va en `realtime/` si se agrega al repo.

## Detector de clips malos (`src/detectar_clips_malos.py`)

Auditoria automatica de los ROIs labiales **96x96** que produce el preprocesamiento
visual. Corre sobre `data/processed/lip_rois/` (lo que efectivamente ve el modelo) y
**no usa MediaPipe**: trabaja directo sobre los pixeles del recorte, asi que es barato
(solo `cv2` + `numpy`). Detecta los tres problemas que mas dañan el fine-tuning:

| Problema | Metrica | Estado |
|---|---|---|
| **Negro / oscuro** (placa, fundido, camara tapada) | `luma_media`, `frac_oscuros` | `drop` |
| **Congelado** (imagen estatica, sin habla real) | `movimiento_global` (diff temporal media) | `drop` |
| **Boca inactiva / tapada** (mano, microfono, mala alineacion, silencio) | `actividad_boca` (std temporal de la region central) + `textura_boca` (varianza de Laplaciano) | `review` |

La oclusion de boca es un proxy **heuristico**: MediaPipe alucina landmarks aunque la
boca este tapada, asi que el preproc no la descarta; aca la cazamos por baja
actividad+textura en la region de la boca y la marcamos `review` (no `drop`), para
mirarla a ojo antes de decidir.

Los umbrales (en `src/detectar_clips_malos.py`) estan **calibrados sobre las 9 fuentes
ya procesadas**: los de `drop` (negro/congelado) son red de seguridad para futuras
fuentes; los de `review` se situan en el piso real de la distribucion para surfacear la
cola dudosa. En este dataset: **1683 `keep`, 21 `review`, 0 `drop`** (ya estaba limpio
de negro/congelado por el filtro de 80% de cara del preproc).

### Uso

```bash
# 1. Auditar todos los ROIs -> escribe el manifest de auditoria
python -m data_cleaning.src.detectar_clips_malos

# Una sola fuente
python -m data_cleaning.src.detectar_clips_malos "<titulo>"

# 2. Materializar el dataset final curado (copia los `keep` -> dataset/)
python -m data_cleaning.src.detectar_clips_malos --materializar
```

Salida:

```text
data/metadata/auditoria_clips_manifest.csv   # un unico manifest: keep/review/drop + metricas + razones
dataset/<titulo>/clip_NNNN.mp4 | .txt         # set final curado (solo `keep`)
dataset/manifest.csv                          # inventario del set curado
```

El paso de materializar **no borra** nada de `data/processed/lip_rois/` (ese es el
insumo crudo y se conserva). Los `review` quedan retenidos: no se copian a `dataset/`
hasta decidirlos a mano.

## Notebook de revision visual

Para mirar a ojo los casos `review` (o sospechosos) con video + texto + razon, la
revision humana arranca en:

```text
data_cleaning/notebooks/01_revision_visual_mediapipe.ipynb
```

El notebook usa MediaPipe para detectar casos sospechosos para VSR:

- 0 caras aunque haya texto;
- mas de una cara;
- boca poco visible;
- boca muy descentrada.

No borra nada. Primero muestra video + texto + razon de revision. Evitar crear CSVs
paralelos por cada experimento; si hace falta guardar una decision, se agrega a un unico
manifest acordado en `data/metadata/`.

## Comparacion de modelos Whisper

El notebook:

```text
data_cleaning/notebooks/02_analisis_whisper_azzaro.ipynb
```

compara `turbo`, `small` y `large` sobre el mismo video, forma clips con timestamps
por palabra y pausas reales, y permite seleccionar manualmente la mejor transcripcion.
Los artefactos finales chicos se versionan en:

```text
data_cleaning/outputs/azzaro_whisper/final/
```

El video original, los modelos descargados y los clips descartados permanecen locales.
