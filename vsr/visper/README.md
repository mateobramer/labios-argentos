# vsr/visper/ — base ViSpeR/TII (código propio versionado)

Código para correr **ViSpeR** (288M, TII), el modelo base que usa la demo
(`demo/demo_web.py`, `demo/infer_server.py`) vía `$VISPER_DIR`
(default `~/Desktop/visper`). Repo upstream: [`YasserdahouML/visper`](https://github.com/YasserdahouML/visper)
(ver `UPSTREAM_README.md`, el README original de ese repo).

El clon en `$VISPER_DIR` **no se versiona** — acá vive una copia del código para no
perderlo entre máquinas (se perdió una vez porque nunca estuvo en git; recuperado
2026-07-11 desde otra máquina).

| Path | Qué es |
|---|---|
| `datamodule/` | `transforms.py`, `data_module.py`, `av_dataset.py`, `samplers.py` — carga y transforma clips |
| `lightning_vsr.py` | `ModelModule` — wrapper PyTorch Lightning del modelo |
| `espnet/` | subset vendoreado de ESPnet que usa `lightning_vsr.py` |
| `conf/` | configs Hydra (`model/visual_backbone/resnet_conformer.yaml`, etc.) |
| `spm/unigram/` | tokenizer sentencepiece (`unigram.model`, `unigram_units.txt`, `unigram.vocab`) |
| `data_prepare/` | `crop_videos.py` + `20words_mean_face.npy` — preprocesamiento upstream (no es el que usa este proyecto, ver `preprocessing/`) |
| `visper_zeroshot.py` | script propio: zero-shot de ViSpeR sobre test-658 (`build_cfg()` — misma config que usa `infer_server.py`) |
| `infer.py`, `utils.py`, `cosine.py`, `WER/` | scripts/utils del repo upstream (evaluación, LR scheduler) |

## Pesos (no versionados — 1.15 GB, supera el límite de 100 MB de GitHub)

`visper_vsr_base.pth` va dentro de `$VISPER_DIR`, NO acá. Fuente verificada
2026-07-11: [`huggingface.co/tiiuae/visper`](https://huggingface.co/tiiuae/visper)
(público, sin login), archivo `visper_vsr_base.pth` (1,153,269,123 bytes exactos).

```bash
curl -L -o ~/Desktop/visper/visper_vsr_base.pth \
  https://huggingface.co/tiiuae/visper/resolve/main/visper_vsr_base.pth
```

## Cómo reconstruir `$VISPER_DIR` en una máquina nueva

```bash
mkdir -p ~/Desktop/visper
cp -r vsr/visper/* vsr/visper/.gitignore ~/Desktop/visper/   # este código
curl -L -o ~/Desktop/visper/visper_vsr_base.pth \
  https://huggingface.co/tiiuae/visper/resolve/main/visper_vsr_base.pth
conda env create -f envs/visper.yml   # si el env `visper` no existe
```
