"""Second-pass ASR para auditar transcripts actuales sin reemplazar ground truth."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO_ROOT / "vsr_models" / "splits" / "splits.csv"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr" / "transcript_second_pass_asr.csv"
DEFAULT_CANDIDATES = REPO_ROOT / "evaluation" / "outputs" / "batch_vsr" / "transcript_cleaning_candidates.csv"

OUTPUT_COLUMNS = [
    "source_id",
    "clip",
    "split",
    "clip_path",
    "current_text",
    "asr2_text",
    "asr2_model",
    "status",
    "reason",
    "asr2_runtime_sec",
]


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalizar_texto(texto: str) -> str:
    return " ".join(str(texto or "").strip().split())


def clip_path(row: dict[str, str], repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data" / "clips" / row["titulo"] / f"{row['clip']}.mp4"


def detectar_backend() -> str | None:
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster-whisper"
    if importlib.util.find_spec("whisper") is not None:
        return "whisper"
    return None


def detectar_device() -> str:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def modelo_whisper(model_name: str) -> str:
    return "turbo" if model_name == "large-v3-turbo" else model_name


class ASR2Runner:
    def __init__(self, backend: str, model_name: str, device: str, compute_type: str, beam_size: int):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.model: Any = None

    def load(self) -> None:
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            compute_type = self.compute_type
            if compute_type == "auto":
                compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
            return
        if self.backend == "whisper":
            import whisper

            self.model = whisper.load_model(modelo_whisper(self.model_name), device=self.device)
            return
        raise ValueError(f"backend no soportado: {self.backend}")

    def transcribe(self, path: Path) -> str:
        if self.backend == "faster-whisper":
            segments, _info = self.model.transcribe(
                str(path),
                beam_size=self.beam_size,
                language="es",
                task="transcribe",
            )
            return normalizar_texto(" ".join(segment.text for segment in segments))
        result = self.model.transcribe(
            str(path),
            language="es",
            task="transcribe",
            fp16=self.device == "cuda",
            verbose=False,
        )
        return normalizar_texto(result.get("text", ""))


def _candidate_keys(candidates_path: Path) -> set[tuple[str, str]]:
    if not candidates_path.exists():
        return set()
    rows = leer_csv(candidates_path)
    return {(row.get("source_id", ""), row.get("clip", "")) for row in rows}


def seleccionar_rows(
    rows: list[dict[str, str]],
    only_suspicious: bool,
    candidates_path: Path,
    max_clips: int | None,
) -> list[dict[str, str]]:
    selected = rows
    if only_suspicious:
        keys = _candidate_keys(candidates_path)
        selected = [row for row in rows if (row["titulo"], row["clip"]) in keys]
    if max_clips is not None and max_clips >= 0:
        selected = selected[:max_clips]
    return selected


def blocked_rows(rows: list[dict[str, str]], reason: str, model_name: str, repo_root: Path) -> list[dict[str, str]]:
    return [
        {
            "source_id": row["titulo"],
            "clip": row["clip"],
            "split": row["split"],
            "clip_path": str(clip_path(row, repo_root)),
            "current_text": row.get("texto", ""),
            "asr2_text": "",
            "asr2_model": model_name,
            "status": "blocked",
            "reason": reason,
            "asr2_runtime_sec": "0.000",
        }
        for row in rows
    ]


def run_second_pass_asr(
    splits_path: Path = DEFAULT_SPLITS,
    output_path: Path = DEFAULT_OUTPUT,
    model_name: str = "large-v3-turbo",
    max_clips: int | None = None,
    only_suspicious: bool = False,
    candidates_path: Path = DEFAULT_CANDIDATES,
    backend: str = "auto",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    rows = seleccionar_rows(leer_csv(splits_path), only_suspicious, candidates_path, max_clips)
    selected_backend = detectar_backend() if backend == "auto" else backend
    if selected_backend is None:
        out_rows = blocked_rows(rows, "blocked_missing_asr_dependency", model_name, repo_root)
        escribir_csv(output_path, out_rows)
        return {
            "status": "blocked",
            "reason": "blocked_missing_asr_dependency",
            "rows": len(out_rows),
            "output": str(output_path),
        }

    selected_device = detectar_device() if device == "auto" else device
    runner = ASR2Runner(selected_backend, model_name, selected_device, compute_type, beam_size)
    try:
        runner.load()
    except Exception as exc:
        out_rows = blocked_rows(rows, f"blocked_asr_model_load_failed:{type(exc).__name__}", model_name, repo_root)
        escribir_csv(output_path, out_rows)
        return {
            "status": "blocked",
            "reason": f"blocked_asr_model_load_failed:{type(exc).__name__}",
            "rows": len(out_rows),
            "output": str(output_path),
        }

    out_rows: list[dict[str, str]] = []
    for row in rows:
        path = clip_path(row, repo_root)
        base = {
            "source_id": row["titulo"],
            "clip": row["clip"],
            "split": row["split"],
            "clip_path": str(path),
            "current_text": row.get("texto", ""),
            "asr2_model": f"{selected_backend}:{model_name}",
        }
        if not path.exists():
            out_rows.append(
                {
                    **base,
                    "asr2_text": "",
                    "status": "blocked",
                    "reason": "blocked_missing_clip",
                    "asr2_runtime_sec": "0.000",
                }
            )
            continue
        started = time.perf_counter()
        try:
            text = runner.transcribe(path)
            runtime = time.perf_counter() - started
            out_rows.append(
                {
                    **base,
                    "asr2_text": text,
                    "status": "ok",
                    "reason": "",
                    "asr2_runtime_sec": f"{runtime:.3f}",
                }
            )
        except Exception as exc:
            runtime = time.perf_counter() - started
            out_rows.append(
                {
                    **base,
                    "asr2_text": "",
                    "status": "failed",
                    "reason": f"asr_exception:{type(exc).__name__}:{exc}",
                    "asr2_runtime_sec": f"{runtime:.3f}",
                }
            )
    escribir_csv(output_path, out_rows)
    ok = sum(1 for row in out_rows if row["status"] == "ok")
    blocked = sum(1 for row in out_rows if row["status"] == "blocked")
    failed = sum(1 for row in out_rows if row["status"] == "failed")
    return {
        "status": "ok" if ok == len(out_rows) else "partial",
        "backend": selected_backend,
        "model": model_name,
        "device": selected_device,
        "rows": len(out_rows),
        "ok": ok,
        "blocked": blocked,
        "failed": failed,
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--only-suspicious", action="store_true")
    ap.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--backend", choices=["auto", "faster-whisper", "whisper"], default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--beam-size", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_second_pass_asr(
        splits_path=args.splits,
        output_path=args.output,
        model_name=args.model,
        max_clips=args.max_clips,
        only_suspicious=args.only_suspicious,
        candidates_path=args.candidates,
        backend=args.backend,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
