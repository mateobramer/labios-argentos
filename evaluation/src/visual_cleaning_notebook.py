"""Tablas de apoyo para el notebook de visual cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.src.experiment_metrics import comparar_experimentos, resumen_por_grupo


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "evaluation" / "outputs" / "visual_cleaning"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
RESULTS_DIR = OUTPUT_DIR / "results"
LLM_DIR = OUTPUT_DIR / "llm_corrector"

MANIFEST_FILES = {
    "original_train": "original_train.csv",
    "original_val": "original_val.csv",
    "original_test": "original_test.csv",
    "visual_cleaned_train": "visual_cleaned_train.csv",
    "visual_cleaned_val": "visual_cleaned_val.csv",
    "visual_cleaned_test_original": "visual_cleaned_test_original.csv",
}

RESULT_FILES = {
    "baseline_original": RESULTS_DIR / "baseline_original_test.csv",
    "visual_cleaned": RESULTS_DIR / "visual_cleaned_test_original.csv",
}

LLM_RESULT_FILES = {
    "baseline_original": LLM_DIR / "baseline_original_llm_corrected.csv",
    "visual_cleaned": LLM_DIR / "visual_cleaned_llm_corrected.csv",
}


def cargar_manifests(manifest_dir: Path = MANIFEST_DIR) -> dict[str, pd.DataFrame]:
    tablas: dict[str, pd.DataFrame] = {}
    for nombre, archivo in MANIFEST_FILES.items():
        path = manifest_dir / archivo
        tablas[nombre] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tablas


def tamanos_manifests(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [{"manifest": nombre, "clips": len(df)} for nombre, df in tablas.items()]
    return pd.DataFrame(rows)


def distribucion_usabilidad(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for nombre, df in tablas.items():
        if df.empty or "training_usability" not in df:
            continue
        counts = df["training_usability"].fillna("(sin valor)").value_counts().to_dict()
        for usability, clips in counts.items():
            rows.append({"manifest": nombre, "training_usability": usability, "clips": clips})
    return pd.DataFrame(rows)


def perdida_por_fuente(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    original = tablas.get("original_train", pd.DataFrame())
    cleaned = tablas.get("visual_cleaned_train", pd.DataFrame())
    if original.empty or cleaned.empty:
        return pd.DataFrame(columns=["source_id", "original_train", "visual_cleaned_train", "excluded"])

    source_col = "source_id" if "source_id" in original else "titulo"
    base = original.groupby(source_col).size().rename("original_train")
    limpio = cleaned.groupby(source_col).size().rename("visual_cleaned_train")
    out = pd.concat([base, limpio], axis=1).fillna(0).astype(int).reset_index()
    out["excluded"] = out["original_train"] - out["visual_cleaned_train"]
    return out.sort_values(["excluded", "original_train"], ascending=False)


def estado_archivos(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for nombre, path in paths.items():
        rows.append({"artifact": nombre, "path": str(path), "exists": path.exists()})
    return pd.DataFrame(rows)


def cargar_resultados_existentes(paths: dict[str, Path] = RESULT_FILES) -> pd.DataFrame:
    tablas = []
    for experiment, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            if "experiment" not in df:
                df["experiment"] = experiment
            tablas.append(df)
    return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()


def resumen_resultados(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = df.to_dict("records")
    if group_col:
        return pd.DataFrame(resumen_por_grupo(rows, group_col))
    return pd.DataFrame([comparar_experimentos(rows)])


def conclusion_automatica(tablas: dict[str, pd.DataFrame], resultados: pd.DataFrame) -> str:
    sizes = {nombre: len(df) for nombre, df in tablas.items()}
    excluidos = sizes.get("original_train", 0) - sizes.get("visual_cleaned_train", 0)
    if resultados.empty:
        return (
            "Pendiente de VM: los manifests estan listos, pero todavia no hay resultados "
            "VSR comparables. La comparacion valida debe usar el test original completo."
        )
    resumen = comparar_experimentos(resultados.to_dict("records"))
    if not resumen["can_conclude"]:
        return "Hay resultados, pero no alcanzan clips suficientes para concluir."
    delta = resumen.get("delta_wer")
    if delta is None:
        return "Hay resultados, pero faltan WER validos para comparar."
    if delta < 0:
        return f"visual_cleaned mejora WER promedio por {-delta:.4f}; revisar grupos y fuentes antes de cerrar."
    return f"visual_cleaned no mejora WER promedio en esta corrida; train excluyo {excluidos} clips."
