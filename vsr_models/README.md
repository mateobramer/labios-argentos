# vsr_models

Fine-tuning del modelo de **lectura de labios (VSR)** al español rioplatense, y **contrato
de datos/splits** para el resto del equipo (ver `ESTRUCTURA_PROYECTO.md`).

## Punto de partida

Heredamos **solo el modelo** de Gimeno
([`david-gimeno/evaluating-end2end-spanish-lipreading`](https://github.com/david-gimeno/evaluating-end2end-spanish-lipreading)):
el checkpoint `vsr-liprtve-si.pth` (Conv3D-ResNet18 → Conformer → CTC/attention, **ESPnet,
tokenizer a nivel carácter**). Ese repo es **solo de evaluación** — el fine-tuning lo
montamos nosotros. Baseline zero-shot a batir: **WER 79.26 / CER 47.20** (`evaluation/`).

## Enfoque (decidido)

**Reusar el modelo + dataloader de Gimeno (ya andan, del zero-shot) y agregarle un training
loop propio** (optimizer + backprop + checkpointing + early stopping). Se mantiene el
checkpoint y el vocabulario char intactos → cero cirugía de pesos.

Por qué y no Auto-AVSR / recipe ESPnet: Auto-AVSR usa SentencePiece (otro vocabulario) →
forzaría re-mapear pesos. Reusar Gimeno es más compatible con los compañeros (ya levantan
el env `vsr-factors` + checkpoint con `evaluation/setup_modelo_gimeno.sh`, cero setup
nuevo) y menos riesgoso.

## Contrato de datos (framework-agnóstico)

`vsr_models/splits/` define los splits **speaker-independent** (ninguna fuente cae en dos
splits). Los genera `src/armar_splits.py` desde la curaduría:

```text
splits/splits.csv            # todo junto
splits/{train,val,test}.csv  # por split
# columnas: split, spk, titulo, clip, n_frames, texto, npz
```

Cada fila apunta a un ROI en `data/processed/lip_rois/<titulo>/<clip>.npz` —un array
`(T, 96, 96)` uint8 gris a 25 fps— + su transcripción limpia (`lower + unidecode + ñ`).
**Cualquier arquitectura** (o capa agéntica) consume esto sin atarse a ESPnet:

```python
import csv, numpy as np
for r in csv.DictReader(open("vsr_models/splits/train.csv")):
    rois = np.load(r["npz"])["rois"]   # (T, 96, 96) uint8
    texto = r["texto"]
    ...
```

| Split | Clips | Hablantes |
|---|---|---|
| train | 4826 | 26 |
| val | 466 | 3 |
| test | 658 | 2 (= fuentes del zero-shot, para comparar) |

Para re-armar los splits (cambiar qué fuentes van a val/test), editar las listas en
`src/armar_splits.py` y correr `python -m vsr_models.src.armar_splits`.

## Estado y próximos pasos

- [x] Splits speaker-independent (`splits/`).
- [ ] Verificar la API de loss del modelo ESPnet en el env `vsr-factors` (de-risk, en la VM).
- [ ] Training loop + fine-tune desde `vsr-liprtve-si.pth` (VM con GPU L4).
- [ ] Evaluar fine-tuned vs zero-shot (79.26 WER) sobre el mismo test.

Los `.npz` (4.2 GB) son locales/gitignored; para entrenar hay que subirlos a la VM.
