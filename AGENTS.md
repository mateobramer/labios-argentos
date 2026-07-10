# AGENTS.md

Guía para trabajar en este repositorio (humanos y agentes).

## Qué es este repo

`labios-argentos` es el **sistema completo** de un proyecto de investigación de
**lectura de labios / VSR en español rioplatense** (Ingeniería en IA, Universidad de
San Andrés): pipeline de datos desde YouTube, fine-tuning de dos familias de modelos
(50M de Gimeno y ViSpeR 288M), evaluación WER/CER, corrector LLM por n-best rescoring,
calibración al hablante con LoRA, y una demo web cerca de tiempo real.

Empezar por el [`README.md`](README.md) (visión de producto y resultados) y
[`docs/ESTRUCTURA.md`](docs/ESTRUCTURA.md) (mapa del repo y flujo de datos).

## Dónde está la verdad

- **Resultados**: [`experiments/README.md`](experiments/README.md) tiene la tabla
  maestra y el índice de los 10 docs de experimentos. [`docs/RESULTS.md`](docs/RESULTS.md)
  es el ledger vivo. **Antes de re-medir algo, fijate si ya está medido.**
- **Decisiones de diseño de la demo**: [`docs/SPEC.md`](docs/SPEC.md) — cada parámetro
  (beam, MPS, VAD, qwen) está justificado con su experimento.
- **Splits congelados**: `vsr_models/splits/` (test-658 y val fijos desde ft03). Una
  comparación solo vale si usa exactamente estos splits y la misma normalización de
  texto: minúsculas, sin acentos (**ñ preservada**), sin puntuación.

## Entornos y máquinas

| Env conda | Para qué |
|---|---|
| `ptt` | demo web / captura (OpenCV + MediaPipe) — es el que lanza `demo/demo_web.py` |
| `visper` | inferencia ViSpeR (PyTorch + ESPnet); requiere el repo en `~/Desktop/visper` |
| `mvsr` | espnet1 vendoreado (mpc001) — corre el 50M/ft05 local vía remap |
| `vsr-factors` | solo en VMs de GCP: entrenamiento del 50M (repo de Gimeno) |

Los entrenamientos corren en **VMs L4 spot de GCP** (proyecto
`visual-speech-recognition-nlp`, imagen `labios-img-visper`, bucket
`gs://labios-argentos-vsr-dataset`) con startup scripts que suben resultados al bucket y
**se autodestruyen**.

## Reglas duras

- **Costos GCP**: toda VM se lanza spot, con auto-destrucción, y se verifica que murió
  (VM **y disco**). No dejar nada corriendo sin monitor.
- **Privacidad**: las grabaciones personales (`~/vsr_personal/`, `~/vsr_contrib/`) y los
  modelos calibrados (`modelos/personal/`) **no se versionan nunca**.
- **No versionar pesados/regenerables**: `.npz`, `.pth`, videos crudos, venvs, clones de
  repos externos (ya está en `.gitignore` — revisarlo antes de commits grandes).
- **YouTube**: no usar cookies del browser ni rotación de IPs; el scraping masivo quedó
  descartado (ver [`experiments/07`](experiments/07_datos_y_scraping.md)). Fuentes
  nuevas: de a una, con los gates de calidad.
- **Git**: el usuario nombra los commits y autoriza los push. Sin `Co-Authored-By`.
- El working copy usa **sparse-checkout** (los ~29k archivos del dataset no están todos
  materializados); si un path trackeado "no existe", revisar `git sparse-checkout list`.

## Estructura y convenciones

- Cada bloque del sistema vive en su propia carpeta, con lógica reutilizable en `src/`
  y su `README.md`. No dejar scripts, datos ni resultados sueltos en la raíz.
- Código y comentarios en **español**, nombres de funciones en español
  (`bajar_video`, `cortar_clips`). Estilo procedural directo, sin frameworks.
- Los directorios bajo `data/` y `dataset/` son **datos generados**: no editarlos a mano.
- La invariante sagrada del dataset es la **alineación video↔texto**: si tocás
  `cortar_clips` o el preproc, el `.txt` debe seguir correspondiendo exacto al clip.
- Experimentos nuevos: documentarlos en `experiments/` (un doc por categoría, actualizar
  la tabla maestra del README de experiments).

## Comandos frecuentes

```bash
# Demo web (UI en http://localhost:8551; --qwen para el corrector, --ckpt para modelo personal)
~/miniconda3/envs/ptt/bin/python demo/demo_web.py

# Pipeline de datos para una fuente nueva (ver claude-videos/README.md)
python descargar_procesar.py "URL_YOUTUBE"
python -m preprocessing.src.preprocesar "<titulo>"
python -m cleaning.visual_quality.src.detectar_clips_malos "<titulo>" [--materializar]

# Calibración al hablante (después de grabar en la UI /calibrar)
bash demo/calibracion/calibrar_entrenar.sh <nombre>

# Scoring del self-test
~/miniconda3/envs/visper/bin/python demo/score_selftest.py --model visper
```
