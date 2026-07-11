"""Filtra clips de musica/alucinacion de una fuente: borra clip_*.mp4+.txt cuyo texto
sea repetitivo (alucinacion tipica de Whisper sobre musica/canto). Loguea lo descartado."""
import os, sys, collections

titulo = sys.argv[1]
d = os.path.join("data", "clips", titulo)
logp = os.path.expanduser("~/music_dropped.log")
dropped = 0
if os.path.isdir(d):
    with open(logp, "a", encoding="utf-8") as log:
        for f in sorted(x for x in os.listdir(d) if x.endswith(".txt")):
            base = f[:-4]
            txt = open(os.path.join(d, f), encoding="utf-8").read().strip()
            w = txt.split()
            if len(w) < 6:
                continue
            uniq = len(set(w)) / len(w)
            grams = [" ".join(w[i:i+3]) for i in range(len(w) - 2)]
            rep = max(collections.Counter(grams).values()) if grams else 0
            # musica/alucinacion: poca variedad lexica O una frase de 3 palabras repetida >=3
            if uniq < 0.35 or rep >= 3:
                for ext in (".mp4", ".txt"):
                    p = os.path.join(d, base + ext)
                    if os.path.exists(p):
                        os.remove(p)
                dropped += 1
                log.write(f"{titulo}\t{base}\tuniq={uniq:.2f}\trep={rep}\t{txt[:100]}\n")
print(f"music_dropped={dropped}")
