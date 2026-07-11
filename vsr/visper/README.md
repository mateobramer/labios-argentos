# vsr/visper/ — base ViSpeR/TII (código propio versionado)

Código propio del equipo para correr **ViSpeR** (288M, TII), el modelo base que usa
la demo (`demo/demo_web.py`, `demo/infer_server.py`) vía `$VISPER_DIR`
(default `~/Desktop/visper`). El clon en `$VISPER_DIR` **no se versiona** (`.gitignore`);
acá vive solo el código propio para no perderlo entre máquinas.

| Path | Qué es |
|---|---|
| `visper_zeroshot.py` | zero-shot de ViSpeR sobre test-658 (`build_cfg()` — misma config que usa `infer_server.py`) |

## Qué falta para que `$VISPER_DIR` funcione (no versionado, ver abajo)

`infer_server.py` y `visper_zeroshot.py` importan `datamodule/`, `lightning_vsr.py`
y leen `conf/model/visual_backbone/resnet_conformer.yaml` + `spm/unigram/unigram.model`
desde la raíz de `$VISPER_DIR`. Son módulos **propios del equipo** (no son del repo
público de mpc001 — se comprobó que ni `mpc001/Visual_Speech_Recognition_for_Multiple_Languages`
ni `mpc001/auto_avsr` tienen esos nombres/estructura exactos). Pendiente traerlos de
la máquina que los tiene.

## Pesos (no versionados — 1.15 GB, supera el límite de 100 MB de GitHub)

`visper_vsr_base.pth` va dentro de `$VISPER_DIR`, NO acá. Fuente verificada
2026-07-11: [`huggingface.co/tiiuae/visper`](https://huggingface.co/tiiuae/visper)
(público, sin login), archivo `visper_vsr_base.pth` (1,153,269,123 bytes exactos).

```bash
curl -L -o ~/Desktop/visper/visper_vsr_base.pth \
  https://huggingface.co/tiiuae/visper/resolve/main/visper_vsr_base.pth
```
