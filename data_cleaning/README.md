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

## Documentacion de trabajo

Cuando implementemos una limpieza o auditoria concreta, dejar aca un README corto o un
notebook con:

- objetivo;
- datos usados;
- criterio de decision;
- salida generada, por ejemplo un manifest en `data/metadata/`.

## Primer flujo

La revision humana arranca en:

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
