"""Helpers de reporte para la auditoria visual offline."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RAIZ_REPO = Path(__file__).resolve().parents[2]


def leer_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def conteo_y_porcentaje(rows: list[dict[str, Any]], columna: str) -> list[dict[str, Any]]:
    total = len(rows) or 1
    conteos = Counter(str(row.get(columna, "")) for row in rows)
    return [
        {"valor": clave, "clips": cantidad, "porcentaje": round(cantidad / total * 100, 2)}
        for clave, cantidad in conteos.most_common()
    ]


def razones_principales(rows: list[dict[str, Any]], columna: str = "used_for_decision_reasons", top: int = 10) -> list[dict[str, Any]]:
    conteos = Counter(
        reason
        for row in rows
        for reason in str(row.get(columna) or row.get("reasons") or "").split(";")
        if reason
    )
    return [{"razon": razon, "clips": cantidad} for razon, cantidad in conteos.most_common(top)]


def resumen_por(rows: list[dict[str, Any]], columna: str) -> list[dict[str, Any]]:
    conteos: dict[tuple[str, str], int] = Counter(
        (str(row.get(columna, "")), str(row.get("decision", ""))) for row in rows
    )
    claves = sorted({clave for clave, _ in conteos})
    salida = []
    for clave in claves:
        total = sum(conteos[(clave, decision)] for decision in ("keep", "review", "drop"))
        item = {columna: clave, "total": total}
        for decision in ("keep", "review", "drop"):
            item[decision] = conteos[(clave, decision)]
            item[f"{decision}_pct"] = round(conteos[(clave, decision)] / max(total, 1) * 100, 2)
        item["retencion_keep_pct"] = item["keep_pct"]
        item["retencion_keep_review_pct"] = round((item["keep"] + item["review"]) / max(total, 1) * 100, 2)
        salida.append(item)
    return salida


def impacto_dataset(rows: list[dict[str, Any]], *, min_clips_por_fuente: int = 5) -> dict[str, Any]:
    total = len(rows)
    decisiones = Counter(row.get("decision", "") for row in rows)
    keep = decisiones.get("keep", 0)
    review = decisiones.get("review", 0)
    por_fuente = resumen_por(rows, "source_id")
    por_split = resumen_por(rows, "split")
    alertas = []
    for item in por_fuente:
        if item["keep"] < min_clips_por_fuente:
            alertas.append(f"fuente_con_pocos_keep:{item['source_id']}={item['keep']}")
    for item in por_split:
        if item["keep"] == 0 and item["total"] > 0:
            alertas.append(f"split_sin_keep:{item['split']}")
    return {
        "clips_totales": total,
        "decisiones": dict(sorted(decisiones.items())),
        "retencion_keep_pct": round(keep / max(total, 1) * 100, 2),
        "retencion_keep_review_pct": round((keep + review) / max(total, 1) * 100, 2),
        "fuentes": len(por_fuente),
        "splits": len(por_split),
        "alertas": alertas,
    }


def escribir_manifest_filtrado(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    decisiones: set[str],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtradas = [row for row in rows if row.get("decision") in decisiones]
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filtradas)
    return output


def escribir_manifest_keep(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    return escribir_manifest_filtrado(rows, output_path, decisiones={"keep"})


def escribir_manifest_keep_review(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    return escribir_manifest_filtrado(rows, output_path, decisiones={"keep", "review"})


def buscar_predicciones_livianas() -> list[dict[str, str]]:
    """Lista archivos de predicciones livianas conocidos, sin asumir que sean joinables."""

    candidatos = []
    for inf_path in sorted(RAIZ_REPO.glob("vsr_models/runs/**/*.inf")) + sorted(RAIZ_REPO.glob("evaluation/outputs/**/*.inf")):
        candidatos.append({"tipo": "inf_ref_hyp", "path": repo_rel(inf_path)})
    for csv_path in sorted(RAIZ_REPO.glob("vsr_models/runs/**/*.csv")) + sorted(RAIZ_REPO.glob("evaluation/outputs/**/*.csv")):
        candidatos.append({"tipo": "csv_posible_metricas", "path": repo_rel(csv_path)})
    for csv_path in sorted(RAIZ_REPO.glob("data/metadata/**/*.csv")):
        if "visual_quality" not in csv_path.name:
            candidatos.append({"tipo": "csv_metadata_posible_mapeo", "path": repo_rel(csv_path)})
    for wer_path in sorted(RAIZ_REPO.glob("vsr_models/runs/**/*.wer")) + sorted(RAIZ_REPO.glob("evaluation/outputs/**/*.wer")):
        candidatos.append({"tipo": "wer_agregado", "path": repo_rel(wer_path)})
    mapeo = RAIZ_REPO / "evaluation" / "outputs" / "mapeo.csv"
    if mapeo.exists():
        candidatos.append({"tipo": "mapeo_clip", "path": repo_rel(mapeo)})
    return candidatos


def cargar_metricas_clip_csv(path: str | Path) -> list[dict[str, Any]]:
    """Carga un CSV futuro de metricas por clip para cruzar con el manifest.

    Formatos aceptados:
    - source_id,clip,wer,cer
    - source_id,clip,split,reference,hypothesis,wer,cer
    """

    path = repo_path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        required = {"source_id", "clip", "wer", "cer"}
        if not required.issubset(fieldnames):
            raise ValueError(
                "CSV de metricas por clip debe incluir source_id,clip,wer,cer "
                "o source_id,clip,split,reference,hypothesis,wer,cer"
            )
        rows = []
        for row in reader:
            rows.append(
                {
                    "source_id": row.get("source_id", ""),
                    "clip": row.get("clip", ""),
                    "split": row.get("split", ""),
                    "wer_clip": float(row.get("wer") or 0.0),
                    "cer_clip": float(row.get("cer") or 0.0),
                    "ref": row.get("reference", ""),
                    "hyp": row.get("hypothesis", ""),
                }
            )
    return rows


def cargar_predicciones_ref_hyp(inf_path: str | Path, mapeo_path: str | Path) -> list[dict[str, Any]]:
    """Carga predicciones por clip desde un archivo ``ref#hyp`` y un mapeo por orden."""

    inf_path = repo_path(inf_path)
    mapeo_path = repo_path(mapeo_path)
    lineas = [line.strip() for line in inf_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with mapeo_path.open("r", encoding="utf-8", newline="") as fh:
        mapeo = list(csv.DictReader(fh))
    n = min(len(lineas), len(mapeo))
    rows = []
    for idx in range(n):
        ref, hyp = partir_ref_hyp(lineas[idx])
        mapa = mapeo[idx]
        rows.append(
            {
                "sampleID": mapa.get("sampleID", ""),
                "source_id": mapa.get("titulo", ""),
                "clip": mapa.get("clip_original", ""),
                "wer_clip": round(error_rate(ref.split(), hyp.split()), 4),
                "cer_clip": round(error_rate(list(ref), list(hyp)), 4),
                "ref": ref,
                "hyp": hyp,
            }
        )
    return rows


def correlacion_quality_vsr(manifest_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_por_clip = {(row["source_id"], row["clip"]): row for row in pred_rows}
    pares = []
    for row in manifest_rows:
        pred = pred_por_clip.get((row.get("source_id", ""), row.get("clip", "")))
        if not pred:
            continue
        try:
            pares.append((float(row["quality_score"]), float(pred["wer_clip"]), float(pred["cer_clip"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(pares) < 3:
        return {"clips_join": len(pares), "quality_vs_wer": "", "quality_vs_cer": ""}
    arr = np.asarray(pares, dtype=np.float32)
    return {
        "clips_join": len(pares),
        "quality_vs_wer": round(float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]), 4),
        "quality_vs_cer": round(float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]), 4),
    }


def comparar_predicciones_por_decision(
    manifest_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_por_clip = {(row["source_id"], row["clip"]): row for row in pred_rows}
    grupos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        pred = pred_por_clip.get((row.get("source_id", ""), row.get("clip", "")))
        if pred:
            grupos[row.get("decision", "")].append(pred)
    salida = []
    for decision, items in sorted(grupos.items()):
        salida.append(
            {
                "decision": decision,
                "clips_join": len(items),
                "wer_promedio": round(float(np.mean([float(x["wer_clip"]) for x in items])), 4),
                "cer_promedio": round(float(np.mean([float(x["cer_clip"]) for x in items])), 4),
            }
        )
    return salida


def formato_predicciones_futuras() -> list[dict[str, str]]:
    return [
        {"columna": "source_id", "descripcion": "titulo/carpeta igual al split y manifest"},
        {"columna": "clip", "descripcion": "clip_NNNN"},
        {"columna": "split", "descripcion": "opcional, split de evaluacion"},
        {"columna": "reference", "descripcion": "opcional, texto esperado"},
        {"columna": "hypothesis", "descripcion": "opcional, hipotesis VSR"},
        {"columna": "wer", "descripcion": "WER por clip, fraccion 0-1"},
        {"columna": "cer", "descripcion": "CER por clip, fraccion 0-1"},
    ]


def generar_previews_manifest(
    rows: list[dict[str, Any]],
    preview_dir: str | Path,
    *,
    por_decision: int = 4,
    por_razon: int = 3,
) -> list[Path]:
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for decision in ("keep", "review", "drop"):
        ejemplos = [row for row in rows if row.get("decision") == decision][:por_decision]
        if not ejemplos:
            continue
        output = preview_dir / f"{decision}_contact_sheet.jpg"
        if escribir_contact_sheet(ejemplos, output):
            outputs.append(output)

    for razon, ejemplos in ejemplos_por_razon(rows, por_razon=por_razon).items():
        output = preview_dir / f"reason_{nombre_seguro(razon)}.jpg"
        if escribir_contact_sheet(ejemplos, output):
            outputs.append(output)
    return outputs


def ejemplos_por_razon(rows: list[dict[str, Any]], *, por_razon: int = 3, top: int = 5) -> dict[str, list[dict[str, Any]]]:
    razones = razones_principales(rows, top=top)
    salida: dict[str, list[dict[str, Any]]] = {}
    for item in razones:
        razon = item["razon"]
        salida[razon] = [
            row
            for row in rows
            if razon in str(row.get("used_for_decision_reasons") or row.get("reasons") or "").split(";")
        ][:por_razon]
    return salida


def escribir_contact_sheet(rows: list[dict[str, Any]], output_path: str | Path, *, frames_por_clip: int = 3) -> bool:
    filas = []
    for row in rows:
        video_path = repo_path(row.get("path_video") or "")
        frames = extraer_miniaturas(video_path, n_frames=frames_por_clip)
        if not frames:
            continue
        frames = [dibujar_cajas(frame, row) for frame in frames]
        etiqueta = (
            f"{row.get('clip','')} {row.get('decision','')} q={row.get('quality_score','')} "
            f"{row.get('input_kind','')} {row.get('audit_confidence','')} "
            f"{(row.get('used_for_decision_reasons') or row.get('reasons',''))[:70]}"
        )
        filas.append(_fila_contacto(frames, etiqueta))
    if not filas:
        return False
    ancho = max(f.shape[1] for f in filas)
    normalizadas = []
    for fila in filas:
        if fila.shape[1] < ancho:
            pad = np.full((fila.shape[0], ancho - fila.shape[1], 3), 245, dtype=np.uint8)
            fila = np.concatenate([fila, pad], axis=1)
        normalizadas.append(fila)
    sheet = np.concatenate(normalizadas, axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)))


def extraer_miniaturas(video_path: str | Path, *, n_frames: int = 3, ancho: int = 160) -> list[np.ndarray]:
    video_path = Path(video_path)
    if not video_path.exists():
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = list(range(n_frames)) if total <= 0 else sorted(set(np.linspace(0, total - 1, min(n_frames, total)).astype(int).tolist()))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            escala = ancho / max(w, 1)
            frame = cv2.resize(frame, (ancho, max(1, int(h * escala))), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def dibujar_cajas(frame_rgb: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    cajas = _cajas_resumen(row)
    if not cajas:
        return frame_rgb
    frame = frame_rgb.copy()
    h, w = frame.shape[:2]
    for caja in cajas:
        x0 = int(float(caja["x"]) * w)
        y0 = int(float(caja["y"]) * h)
        x1 = int((float(caja["x"]) + float(caja["w"])) * w)
        y1 = int((float(caja["y"]) + float(caja["h"])) * h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 60, 40), 2)
    return frame


def partir_ref_hyp(linea: str) -> tuple[str, str]:
    if "#" not in linea:
        return linea.strip(), ""
    ref, hyp = linea.split("#", 1)
    return ref.strip(), hyp.strip()


def error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, token_ref in enumerate(ref, start=1):
        cur = [i]
        for j, token_hyp in enumerate(hyp, start=1):
            costo = 0 if token_ref == token_hyp else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + costo))
        prev = cur
    return prev[-1] / len(ref)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return RAIZ_REPO / path


def repo_rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(RAIZ_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def nombre_seguro(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return value[:80] or "razon"


def _cajas_resumen(row: dict[str, Any]) -> list[dict[str, float]]:
    raw = row.get("face_boxes_summary") or ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    for item in data:
        boxes = item.get("boxes") or []
        if boxes:
            return boxes[:4]
    return []


def _fila_contacto(frames: list[np.ndarray], etiqueta: str) -> np.ndarray:
    alto = max(frame.shape[0] for frame in frames)
    normalizados = []
    for frame in frames:
        if frame.shape[0] < alto:
            pad = np.full((alto - frame.shape[0], frame.shape[1], 3), 245, dtype=np.uint8)
            frame = np.concatenate([frame, pad], axis=0)
        normalizados.append(frame)
    tira = np.concatenate(normalizados, axis=1)
    label = np.full((42, tira.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(label, etiqueta[:120], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([label, tira], axis=0)
