"""Auditoria de alineacion clip<->texto por re-transcripcion de audio.

Re-transcribe el AUDIO de cada `data/clips/<titulo>/clip_NNNN.mp4` con Whisper y lo
compara contra:
  - su propio `.txt` (lo que el pipeline le asigno);
  - el `.txt` del clip anterior y del siguiente.

Sirve para cazar los dos problemas de calidad detectados en el dataset:
  1. **Desfase temporal** (clip<->texto corrido): el audio del clip matchea MEJOR el
     texto del vecino que el propio -> `drift_prev` / `drift_next`. Causa raiz: cortar
     con timestamps de segmento de Whisper (sin word-timestamps), que derivan en habla
     continua. Ya corregido en descargar_procesar.py (corte por palabra/pausa).
  2. **Error de transcripcion**: el audio coincide con su propio texto pero con WER alto
     (Whisper escucho mal palabras) -> `texto_dudoso`. Mismo eslabon: el modelo Whisper.

Es complementario a `detectar_clips_malos.py` (que mira pixeles del ROI, no el audio)
y a `whisper_model_comparison.py` (que compara modelos de Whisper).

Uso (desde la raiz del repo, con un entorno que tenga openai-whisper):
    python -m data_cleaning.src.auditar_alineacion                 # todas las fuentes
    python -m data_cleaning.src.auditar_alineacion "<titulo>"      # una fuente
    WHISPER_MODEL=large-v3 python -m data_cleaning.src.auditar_alineacion "<titulo>"

Salida:
    data/metadata/auditoria_alineacion_manifest.csv
"""
import csv
import os
import re
import sys

from unidecode import unidecode

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIPS_DIR = os.path.join(RAIZ, "data", "clips")
MANIFEST = os.path.join(RAIZ, "data", "metadata", "auditoria_alineacion_manifest.csv")

MODELO_NOMBRE = os.environ.get("WHISPER_MODEL", "turbo")
# Umbrales (fraccion de palabras del audio presentes en el texto comparado).
OV_MIN_PROPIO = 0.50   # debajo de esto, el propio texto no explica el audio
MARGEN_VECINO = 0.15   # un vecino "gana" si supera al propio por este margen


def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return unidecode(texto.replace("ñ", "ENIE")).replace("ENIE", "ñ").strip()


def palabras(s):
    return set(limpiar(s).split())


def overlap(audio, ref):
    """Fraccion de palabras del AUDIO que aparecen en ref (0 si audio vacio)."""
    wa, wr = palabras(audio), palabras(ref)
    return len(wa & wr) / len(wa) if wa else 0.0


def leer_txt(carpeta, idx):
    p = os.path.join(carpeta, f"clip_{idx:04d}.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def veredicto(o, p, x):
    if o < OV_MIN_PROPIO and (p > o + MARGEN_VECINO or x > o + MARGEN_VECINO):
        return "drift_prev" if p >= x else "drift_next"
    if o < OV_MIN_PROPIO:
        return "texto_dudoso"
    return "ok"


def auditar_fuente(titulo, modelo, filas):
    carpeta = os.path.join(CLIPS_DIR, titulo)
    clips = sorted(f for f in os.listdir(carpeta) if f.endswith(".mp4"))
    print(f"\n=== {titulo} ({len(clips)} clips) ===")
    for nombre in clips:
        idx = int(re.search(r"(\d+)", nombre).group(1))
        r = modelo.transcribe(os.path.join(carpeta, nombre), language="es",
                              fp16=False, verbose=False)
        audio = limpiar(r["text"])
        propio = leer_txt(carpeta, idx)
        o = overlap(audio, propio)
        p = overlap(audio, leer_txt(carpeta, idx - 1))
        x = overlap(audio, leer_txt(carpeta, idx + 1))
        v = veredicto(o, p, x)
        if v != "ok":
            print(f"  [{v}] clip_{idx:04d}  propio={o:.2f} prev={p:.2f} next={x:.2f}")
        filas.append([titulo, f"clip_{idx:04d}", v, f"{o:.3f}", f"{p:.3f}", f"{x:.3f}",
                      audio, propio])


def main():
    if not os.path.isdir(CLIPS_DIR):
        print(f"ERROR: no existe '{CLIPS_DIR}'.")
        sys.exit(1)

    titulos = [sys.argv[1]] if len(sys.argv) >= 2 else sorted(
        d for d in os.listdir(CLIPS_DIR) if os.path.isdir(os.path.join(CLIPS_DIR, d)))

    import whisper
    print(f"Cargando Whisper {MODELO_NOMBRE}...")
    modelo = whisper.load_model(MODELO_NOMBRE)

    filas = []
    for titulo in titulos:
        auditar_fuente(titulo, modelo, filas)

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["titulo", "clip", "veredicto", "ov_propio", "ov_prev", "ov_next",
                    "audio_transcripto", "texto_asignado"])
        w.writerows(filas)

    n_drift = sum(1 for r in filas if r[2].startswith("drift"))
    n_dudoso = sum(1 for r in filas if r[2] == "texto_dudoso")
    print(f"\nTotal: {len(filas)} clips | desfasados: {n_drift} | texto dudoso: {n_dudoso}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
