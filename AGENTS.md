# AGENTS.md

Guía para trabajar en este repositorio.

## Qué es este repo

`labios-argentos` es la **herramienta de recolección y preprocesamiento de datos** para
un proyecto de investigación de **lectura de labios / reconocimiento visual del habla
(VSR) en español rioplatense**. Su único propósito es construir un dataset propio a
partir de videos de YouTube: descarga el video, lo transcribe, arma un corpus de texto
y lo corta en clips cortos de video alineados con su transcripción.

Forma parte del proyecto académico documentado en `../survey-nlp` (paper en
`survey-nlp/paper/main.tex`): *"Lectura de labios en tiempo real para español
rioplatense: destilación causal de Auto-AVSR con corrección mediante un LLM"*
(Ingeniería en IA, Universidad de San Andrés). El paper propone afinar Auto-AVSR al
español usando LIP-RTVE **más un conjunto propio recolectado de YouTube en
rioplatense** — ese conjunto propio es lo que produce este repo.

## Pipeline

Todo vive en un único script: `descargar_procesar.py`. Se ejecuta con:

```
python descargar_procesar.py URL_YOUTUBE
```

Pasos (funciones en orden):

1. `bajar_video(url)` — usa **yt-dlp** para obtener el título y descargar el `.mp4`.
   Crea `data/videos/<titulo>/`.
2. `transcribir(video_path, carpeta)` — transcribe con **Whisper** (modelo `small`,
   `language="es"`). Cachea el resultado en
   `data/corpus/<titulo>/transcripcion.json`; si ya existe, lo reutiliza.
3. `guardar_corpus(resultado, carpeta)` — vuelca el texto limpio (segmentos de ≥3
   palabras) a `data/corpus/<titulo>/corpus.txt`, una línea por segmento.
4. `cortar_clips(video_path, resultado, carpeta)` — agrupa segmentos en bloques de
   ~3–10 s y usa **ffmpeg** para cortar clips. Por cada clip genera
   `data/clips/<titulo>/clip_NNNN.mp4` + `clip_NNNN.txt` (transcripción limpia del clip).

`limpiar(texto)`: minúsculas, quita puntuación, translitera acentos con `unidecode`
pero **preserva la ñ** (truco del placeholder `ENIE`).

## Estructura de salida

- `data/videos/<titulo>/` — el `.mp4` original (y a veces `.info.json` de yt-dlp).
- `data/corpus/<titulo>/` — `transcripcion.json` (salida cruda de Whisper) + `corpus.txt`.
- `data/clips/<titulo>/` — pares `clip_NNNN.mp4` / `clip_NNNN.txt`.
- `data/metadata/` — inventario de fuentes y hablantes.

Hay 9 fuentes ya procesadas (~1987 clips). Estos directorios son **datos generados**, no
código; no los edites a mano.

## Dependencias

Hay `requirements.txt` para las dependencias de Python. Además se necesita:

- Python: `openai-whisper`, `unidecode`
- Binarios externos en el PATH: `yt-dlp`, `ffmpeg`

⚠️ **Ojo con el PATH de ffmpeg**: el script intenta sumar una ruta típica de winget en
Windows solo si existe; en otras máquinas ffmpeg debe estar disponible globalmente.

## Convenciones

- Código y comentarios en **español**; nombres de funciones en español
  (`bajar_video`, `cortar_clips`).
- Script procedural simple, sin tests ni framework. Mantener ese estilo directo.
- Los títulos de YouTube se sanitizan a ≤50 chars para usarlos como nombre de carpeta
  (`nombre_carpeta`), así que el mismo título mapea consistentemente entre
  `data/videos/`, `data/corpus/` y `data/clips/`.

## Contexto útil

El objetivo final del dataset es entrenar/afinar un modelo de VSR causal y liviano, por
lo que importa la **alineación video↔texto** de cada clip. Si tocás `cortar_clips`,
cuidá que el `.txt` siga correspondiendo exactamente al segmento de video cortado.
