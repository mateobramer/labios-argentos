# labios-argentos

Herramienta de **recolección y preprocesamiento de datos** para un proyecto de
investigación de **lectura de labios / reconocimiento visual del habla (VSR) en español
rioplatense**.

A partir de un video de YouTube, este repo lo descarga, lo transcribe con Whisper, arma
un corpus de texto y lo corta en **clips cortos de video alineados con su
transcripción** — el insumo para entrenar/afinar un modelo de VSR.

---

## Contexto del proyecto

Este repositorio forma parte del proyecto académico documentado en `../survey-nlp`
(paper *"Lectura de labios en tiempo real para español rioplatense: destilación causal
de Auto-AVSR con corrección mediante un LLM"*, Ingeniería en IA, Universidad de San
Andrés).

El estado del arte en VSR está casi exclusivamente en inglés y en modelos no causales
(offline). La propuesta del proyecto:

1. Partir de **Auto-AVSR** como modelo *teacher*.
2. **Afinarlo al español** usando LIP-RTVE (TV española) **+ un dataset propio de
   YouTube en rioplatense** — ese dataset propio es lo que produce este repo.
3. **Destilarlo** a un *student* causal y liviano (ResNet-3D + Conformer causal + CTC)
   que solo observa cuadros pasados, apto para tiempo real.
4. Sumar un **LLM local** (qwen3:4b vía Ollama) que corrige el texto crudo sobre la
   marcha (p. ej. `"vos tienes rason che"` -> `"vos tenés razón, che"`).

Pregunta de investigación: **¿cuánto se degrada el WER (Word Error Rate) al comprimir y
adaptar Auto-AVSR al rioplatense?**

La vista completa del pipeline de trabajo, incluyendo datos, VSR, tiempo real, corrector
LM y feedback loop de correcciones, esta en [`PIPELINE_PROYECTO.md`](PIPELINE_PROYECTO.md).

---

## Alcance de *este* repositorio

Este repo cubre **solo la etapa de datos** del proyecto:

- Descarga de videos de YouTube.
- Transcripción automática (Whisper).
- Generación de corpus de texto.
- Corte en clips video<->texto alineados.

El entrenamiento del modelo VSR, la destilación y el corrector LLM **viven fuera de este
repo** (son parte del proyecto mayor, no del pipeline de datos).

---

## Cómo funciona el pipeline

Todo está en un único script, `descargar_procesar.py`:

```bash
python descargar_procesar.py "URL_DE_YOUTUBE"
```

Pasos:

1. **`bajar_video`** — `yt-dlp` descarga el `.mp4` -> `data/videos/<titulo>/`.
2. **`transcribir`** — Whisper (modelo `turbo` por defecto, español) transcribe.
   Se cachea en `data/corpus/<titulo>/transcripcion.json` (si ya existe, se reutiliza).
3. **`guardar_corpus`** — vuelca texto limpio (segmentos de >=3 palabras) a
   `data/corpus/<titulo>/corpus.txt`.
4. **`cortar_clips`** — agrupa segmentos en bloques de ~3-10 s y con `ffmpeg` genera
   `data/clips/<titulo>/clip_NNNN.mp4` + `clip_NNNN.txt` (transcripción del clip).

### Estructura de salida

```
data/videos/<titulo>/   # .mp4 original descargado
data/corpus/<titulo>/   # transcripcion.json (Whisper) + corpus.txt
data/clips/<titulo>/    # pares clip_NNNN.mp4 / clip_NNNN.txt
data/metadata/          # inventario de fuentes y hablantes
```

---

## Instalación

Requiere **Python 3.9+**. `yt-dlp` y un binario local de `ffmpeg` se instalan con
`requirements.txt`; si ya tenés `ffmpeg` global en el PATH, el script usa ese.

```bash
# Dependencias de Python
pip install -r requirements.txt

# Opcional: ffmpeg del sistema, si se prefiere al binario de imageio-ffmpeg.
#   macOS (Homebrew):
brew install ffmpeg
#   Linux (Debian/Ubuntu):
#   sudo apt install ffmpeg
#   Windows: winget install Gyan.FFmpeg
```

El script verifica al inicio que `yt-dlp` y alguna variante de `ffmpeg` estén
disponibles y avisa si faltan.

Por defecto usa Whisper `turbo`, que suele dar mejor precisión que `small` con
buena velocidad. Para forzar otro modelo:

```bash
WHISPER_MODEL=small python descargar_procesar.py URL_YOUTUBE
```

---

## Estado actual

- Pipeline end-to-end funcionando (descarga -> transcripción -> corpus -> clips).
- Varios videos procesados (reacciones de fútbol, *story time* y conferencias en
  rioplatense).
- Cacheo de transcripciones para no re-procesar.

## Próximos pasos

- [ ] **Ampliar el dataset**: más videos y más variedad de hablantes/acentos
      rioplatenses para cubrir voseo, léxico y fonética local.
- [ ] **Control de calidad** de la alineación video<->texto (revisar clips donde Whisper
      falla o el corte desfasa el texto).
- [ ] **Filtrar clips** sin rostro visible o con cara fuera de cuadro (clave para VSR).
- [ ] **Estandarizar el formato de salida** al esperado por el pipeline de
      entrenamiento (LIP-RTVE / Auto-AVSR): resolución, fps, normalización del texto.
- [ ] **Detección y recorte de la región labial** (MediaPipe, 96x96) — hoy se
      guarda el frame completo; el modelo necesita el crop de labios.
- [ ] **Metadatos por clip** (duración, hablante, fuente) para armar los splits
      train/test.
- [ ] Definir un **conjunto de test rioplatense** separado para la evaluación de WER.

---

## Convenciones

- Código y comentarios en **español**.
- Script procedural simple, sin framework ni tests.
- Los directorios `data/videos/`, `data/corpus/` y `data/clips/` son **datos
  generados**: no editar a mano.
- Los títulos de YouTube se sanitizan a <=50 chars como nombre de carpeta, de modo que el
  mismo video mapea consistentemente entre las tres carpetas.
- Un video crudo supera los 100 MB (límite de GitHub), por eso está en `.gitignore`; se
  regenera con el script.
