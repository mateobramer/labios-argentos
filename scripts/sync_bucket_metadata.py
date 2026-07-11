"""Sincroniza metadata liviana (manifests/reports/metadata) del bucket clean-v1 al repo.

No baja video, audio, ROIs ni pesos: filtra por extension (lista de rechazo) y por
prefijo (solo las carpetas de metadata, nunca clips_mp4/clips_with_audio/rois_npz/
source_videos/source_audio/raw_or_source ni spanish_general).

Uso:
    python scripts/sync_bucket_metadata.py                 # sincroniza
    python scripts/sync_bucket_metadata.py --dry-run        # solo lista, no copia nada
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_BUCKET = "gs://labios-argentos-vsr-clean-v1"
ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "data_release" / "bucket_metadata"

# Unicos prefijos que se sincronizan. Todo lo demas del bucket (clips, source
# videos/audio, ROIs, spanish_general) queda afuera por diseno: no aparece en
# esta lista, no se lista ni se copia.
SYNC_PREFIXES = [
    "manifests",
    "reports",
    "argentina/new_discovery/metadata",
    "argentina/new_discovery/manifests",
    "argentina/combined/manifests",
]

# Cualquier objeto con una de estas extensiones se descarta, incluso si cae
# dentro de un prefijo sincronizado (defensa en profundidad).
HEAVY_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".wav", ".mp3",
    ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip",
}


def tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
    if not found:
        raise RuntimeError(
            f"No se encontro '{name}' en PATH. Instalar Google Cloud SDK: "
            "https://cloud.google.com/sdk/docs/install"
        )
    return found


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def list_objects(gcloud: str, prefix: str) -> list[str]:
    """Lista objetos (no carpetas) bajo un prefijo del bucket, recursivo."""
    result = subprocess.run(
        [gcloud, "storage", "ls", "-r", f"{SOURCE_BUCKET}/{prefix}/**"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "matched no objects" in stderr.lower() or "not found" in stderr.lower():
            return []
        raise RuntimeError(f"gcloud storage ls fallo para {prefix}: {stderr}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # gcloud storage ls -r con ** solo devuelve objetos reales (no imprime
    # marcadores de carpeta), pero por las dudas filtramos los que terminan en "/".
    return [line for line in lines if not line.endswith("/")]


def is_heavy(object_uri: str) -> bool:
    suffix = Path(object_uri).suffix.lower()
    return suffix in HEAVY_EXTENSIONS


def copy_object(gcloud: str, object_uri: str, dest_root: Path) -> Path:
    rel_path = object_uri[len(SOURCE_BUCKET) + 1 :]
    dest_path = dest_root / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [gcloud, "storage", "cp", object_uri, str(dest_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcloud storage cp fallo para {object_uri}: {result.stderr.strip()}")
    return dest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo lista lo que se copiaria, no descarga nada.",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST_ROOT,
        help=f"Carpeta destino (default: {DEST_ROOT.relative_to(ROOT)}).",
    )
    args = parser.parse_args()

    gcloud = tool("gcloud")

    print(f"Bucket origen: {SOURCE_BUCKET}")
    print(f"Destino local: {args.dest}")
    print(f"Prefijos a sincronizar: {', '.join(SYNC_PREFIXES)}")
    print()

    per_prefix_counts: dict[str, int] = {}
    per_prefix_bytes: dict[str, int] = {}
    rejected_heavy: list[str] = []
    copied = 0
    total_bytes = 0

    for prefix in SYNC_PREFIXES:
        objects = list_objects(gcloud, prefix)
        kept = []
        for obj in objects:
            if is_heavy(obj):
                rejected_heavy.append(obj)
                continue
            kept.append(obj)

        per_prefix_counts[prefix] = len(kept)
        prefix_bytes = 0

        for obj in kept:
            if args.dry_run:
                print(f"[dry-run] {obj}")
                continue
            dest_path = copy_object(gcloud, obj, args.dest)
            size = dest_path.stat().st_size
            prefix_bytes += size
            total_bytes += size
            copied += 1

        per_prefix_bytes[prefix] = prefix_bytes
        print(f"{prefix}: {len(kept)} archivos" + ("" if args.dry_run else f", {human_size(prefix_bytes)}"))

    print()
    print("=== Resumen ===")
    for prefix in SYNC_PREFIXES:
        count = per_prefix_counts.get(prefix, 0)
        size = per_prefix_bytes.get(prefix, 0)
        print(f"  {prefix:45s} {count:5d} archivos  {human_size(size)}")

    if rejected_heavy:
        print()
        print(f"Rechazados por extension pesada ({len(rejected_heavy)}):")
        for obj in rejected_heavy:
            print(f"  - {obj}")

    print()
    if args.dry_run:
        print("Dry-run: no se copio nada.")
    else:
        print(f"Total copiado: {copied} archivos, {human_size(total_bytes)}")
        print("No se bajaron videos, audios, npz ni pesos (bloqueados por extension/prefijo).")
        print("No se escribio nada en el bucket (solo lecturas: ls + cp de bucket a disco).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
