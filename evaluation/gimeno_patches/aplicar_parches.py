"""
Aplica los parches minimos al repo de Gimeno (`evaluating-end2end-spanish-lipreading`)
para poder evaluar nuestra base "Rioplatense" con su modelo, sin tocar su pipeline.

Son tres cambios, todos idempotentes:
  1. src/MyDataset.py  -> registrar "Rioplatense" con delimiter=5 (como LIP-RTVE).
  2. src/MyDataset.py  -> limpiar el texto de "Rioplatense" igual que LIP-RTVE
                          (lower + quita puntuacion + unidecode preservando ñ).
  3. vsr_main.py       -> usar mean/std de LIP-RTVE (0.491, 0.166) para "Rioplatense",
                          porque cargamos su checkpoint (fps queda 25 por default).

Uso (en la VM):
    python aplicar_parches.py ~/evaluating-end2end-spanish-lipreading
"""

import os
import sys

PARCHES = [
    # (archivo, viejo, nuevo)
    (
        "src/MyDataset.py",
        'if self.database in ["LIP-RTVE", "Multilingual-TEDx-Spanish", "CMU-MOSEAS-Spanish"]:',
        'if self.database in ["LIP-RTVE", "Multilingual-TEDx-Spanish", "CMU-MOSEAS-Spanish", "Rioplatense"]:',
    ),
    (
        "src/MyDataset.py",
        'if self.database in ["LIP-RTVE", "VLRF"]:',
        'if self.database in ["LIP-RTVE", "VLRF", "Rioplatense"]:',
    ),
    (
        "vsr_main.py",
        "    else:\n        (mean, std) = (0.421, 0.165)",
        '    elif args.database == "Rioplatense":\n        (mean, std) = (0.491, 0.166)\n    else:\n        (mean, std) = (0.421, 0.165)',
    ),
]


def main():
    repo = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/evaluating-end2end-spanish-lipreading")
    if not os.path.isdir(repo):
        print(f"ERROR: no existe el repo '{repo}'")
        sys.exit(1)

    for rel, viejo, nuevo in PARCHES:
        path = os.path.join(repo, rel)
        with open(path, encoding="utf-8") as f:
            contenido = f.read()
        if nuevo in contenido:
            print(f"[ya aplicado] {rel}")
            continue
        if viejo not in contenido:
            print(f"[NO MATCH] {rel}: no encontre el fragmento esperado. Revisar manualmente.")
            print(f"           buscaba: {viejo!r}")
            sys.exit(2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(contenido.replace(viejo, nuevo, 1))
        print(f"[parcheado] {rel}")

    print("Parches OK. La base 'Rioplatense' ya es evaluable con vsr_main.py.")


if __name__ == "__main__":
    main()
