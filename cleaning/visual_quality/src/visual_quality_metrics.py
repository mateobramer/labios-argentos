"""Metricas visuales livianas para auditar clips VSR offline.

La auditoria prioriza el input real del VSR (ROIs ``.npz``). Cuando esos archivos no
estan disponibles, puede caer a videos, pero las senales quedan marcadas como proxy.
No usa modelos pesados ni descarga pesos: el detector de caras es Haar de OpenCV y se
trata solo como alerta liviana.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrames:
    """Frames grises muestreados y metadata minima de la entrada."""

    frames: np.ndarray
    n_frames_total: int
    fps: float
    input_kind: str
    path: Path | None
    frames_esperados: int = 0


def score_rango(valor: float, malo: float, bueno: float) -> float:
    """Normaliza un valor donde mas alto es mejor."""

    if bueno <= malo:
        return 0.0
    return float(np.clip((valor - malo) / (bueno - malo), 0.0, 1.0))


def score_intervalo(valor: float, bajo_malo: float, bajo_bueno: float, alto_bueno: float, alto_malo: float) -> float:
    """Score alto cuando ``valor`` cae dentro de un intervalo sano."""

    if valor < bajo_bueno:
        return score_rango(valor, bajo_malo, bajo_bueno)
    if valor <= alto_bueno:
        return 1.0
    if valor >= alto_malo:
        return 0.0
    return float(np.clip((alto_malo - valor) / (alto_malo - alto_bueno), 0.0, 1.0))


def indices_muestra(n_frames: int, max_frames: int) -> list[int]:
    if n_frames <= 0 or max_frames <= 0:
        return []
    n = min(n_frames, max_frames)
    return sorted(set(np.linspace(0, n_frames - 1, n).astype(int).tolist()))


def cargar_frames(path: str | Path, *, max_frames: int = 80, input_kind: str | None = None) -> VideoFrames:
    """Carga frames grises desde ``.npz`` o video, muestreados uniformemente."""

    path = Path(path)
    if path.suffix.lower() == ".npz":
        return _cargar_frames_npz(path, max_frames=max_frames, input_kind=input_kind or "roi_npz")
    return _cargar_frames_video(path, max_frames=max_frames, input_kind=input_kind or "video")


def metricas_calidad_frames(video: VideoFrames) -> dict[str, Any]:
    """Calcula senales de calidad sobre frames grises."""

    frames = video.frames
    esperados = video.frames_esperados or min(video.n_frames_total, len(frames))
    frame_read_ratio = len(frames) / max(esperados, 1)

    if frames.size == 0 or len(frames) == 0:
        return {
            "n_frames": video.n_frames_total,
            "frames_muestreados": 0,
            "frames_esperados": esperados,
            "frame_read_ratio": 0.0,
            "missing_frame_score": 0.0,
            "luma_media": "",
            "luma_p01": "",
            "luma_p99": "",
            "frac_oscuros": "",
            "contraste_medio": "",
            "blur_laplaciano": "",
            "movimiento_global": "",
            "actividad_boca": "",
            "textura_boca": "",
            "contraste_boca": "",
            "mouth_inactive_frame_ratio": "",
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "blur_score": 0.0,
            "motion_score": 0.0,
            "mouth_activity_score": 0.0,
            "mouth_texture_score": 0.0,
            "mouth_visibility_score": 0.0,
            "scene_cut_score": 1.0,
            "scene_cut_count": 0,
            "scene_cut_max_diff": "",
            "scene_cut_positions": "",
            "video_legible": False,
        }

    luma_por_frame = frames.mean(axis=(1, 2))
    luma_media = float(luma_por_frame.mean())
    luma_p01 = float(np.percentile(luma_por_frame, 1))
    luma_p99 = float(np.percentile(luma_por_frame, 99))
    frac_oscuros = float((luma_por_frame < 25.0).mean())
    contraste_medio = float(frames.std(axis=(1, 2)).mean())

    if len(frames) >= 2:
        diffs = np.abs(np.diff(frames, axis=0))
        movimiento_global = float(diffs.mean())
    else:
        movimiento_global = 0.0

    boca = recorte_boca_proxy(frames, video.input_kind)
    actividad_boca = float(boca.std(axis=0).mean()) if len(boca) >= 2 else 0.0
    textura_boca = float(np.mean([cv2.Laplacian(f, cv2.CV_32F).var() for f in boca])) if len(boca) else 0.0
    contraste_boca = float(boca.std(axis=(1, 2)).mean()) if len(boca) else 0.0
    if len(boca) >= 2:
        diff_boca = np.abs(np.diff(boca, axis=0)).mean(axis=(1, 2))
        mouth_inactive_frame_ratio = float((diff_boca < 1.5).mean())
    else:
        mouth_inactive_frame_ratio = 1.0

    blur_laplaciano = float(np.mean([cv2.Laplacian(f, cv2.CV_32F).var() for f in frames]))
    scene = metricas_cortes_escena(frames)

    brightness_score = score_intervalo(luma_media, 35.0, 65.0, 210.0, 240.0)
    contrast_score = score_rango(contraste_medio, 12.0, 45.0)
    blur_score = score_rango(blur_laplaciano, 20.0, 120.0)
    motion_score = score_rango(movimiento_global, 1.0, 5.0)
    mouth_activity_score = score_rango(actividad_boca, 4.0, 14.0)
    mouth_texture_score = score_rango(textura_boca, 12.0, 45.0)
    mouth_contrast_score = score_rango(contraste_boca, 8.0, 30.0)
    mouth_luma_score = score_intervalo(float(boca.mean()) if boca.size else 0.0, 30.0, 55.0, 220.0, 245.0)
    mouth_visibility_score = float(np.mean([mouth_texture_score, mouth_contrast_score, mouth_luma_score]))

    return {
        "n_frames": video.n_frames_total,
        "frames_muestreados": int(len(frames)),
        "frames_esperados": int(esperados),
        "frame_read_ratio": round(float(frame_read_ratio), 4),
        "missing_frame_score": round(float(frame_read_ratio), 4),
        "luma_media": round(luma_media, 3),
        "luma_p01": round(luma_p01, 3),
        "luma_p99": round(luma_p99, 3),
        "frac_oscuros": round(frac_oscuros, 4),
        "contraste_medio": round(contraste_medio, 3),
        "blur_laplaciano": round(blur_laplaciano, 3),
        "movimiento_global": round(movimiento_global, 3),
        "actividad_boca": round(actividad_boca, 3),
        "textura_boca": round(textura_boca, 3),
        "contraste_boca": round(contraste_boca, 3),
        "mouth_inactive_frame_ratio": round(mouth_inactive_frame_ratio, 4),
        "brightness_score": round(brightness_score, 4),
        "contrast_score": round(contrast_score, 4),
        "blur_score": round(blur_score, 4),
        "motion_score": round(motion_score, 4),
        "mouth_activity_score": round(mouth_activity_score, 4),
        "mouth_texture_score": round(mouth_texture_score, 4),
        "mouth_visibility_score": round(mouth_visibility_score, 4),
        "scene_cut_score": scene["scene_cut_score"],
        "scene_cut_count": scene["scene_cut_count"],
        "scene_cut_max_diff": scene["scene_cut_max_diff"],
        "scene_cut_positions": scene["scene_cut_positions"],
        "video_legible": True,
    }


def recorte_boca_proxy(frames: np.ndarray, input_kind: str) -> np.ndarray:
    """Devuelve una region de boca aproximada."""

    _, h, w = frames.shape
    if "roi" in input_kind:
        y0, y1 = int(h * 0.30), int(h * 0.78)
        x0, x1 = int(w * 0.18), int(w * 0.82)
    else:
        # Fallback debil: zona central-inferior del cuadro, no boca real.
        y0, y1 = int(h * 0.42), int(h * 0.82)
        x0, x1 = int(w * 0.25), int(w * 0.75)
    return frames[:, max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]


def metricas_cortes_escena(frames: np.ndarray) -> dict[str, Any]:
    """Detecta discontinuidades visuales bruscas dentro del clip."""

    if len(frames) < 3:
        return {"scene_cut_score": 0.0, "scene_cut_count": 0, "scene_cut_max_diff": 0.0, "scene_cut_positions": ""}

    reducidos = np.stack([cv2.resize(f, (48, 48), interpolation=cv2.INTER_AREA) for f in frames], axis=0)
    diffs = np.abs(np.diff(reducidos, axis=0)).mean(axis=(1, 2))
    mediana = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - mediana))) + 1e-6
    pico = float(np.max(diffs))
    z_robusto = (pico - mediana) / (1.4826 * mad)
    umbral_corte = max(35.0, mediana + 6.0 * 1.4826 * mad)
    posiciones = [int(i + 1) for i, diff in enumerate(diffs) if diff >= umbral_corte]

    riesgo_por_pico = score_rango(pico, 28.0, 65.0)
    riesgo_por_z = score_rango(z_robusto, 5.0, 11.0) if pico >= 28.0 else 0.0
    riesgo_por_cantidad = score_rango(len(posiciones), 0.0, 3.0)
    score = float(np.clip(max(riesgo_por_pico, riesgo_por_z, riesgo_por_cantidad), 0.0, 1.0))
    return {
        "scene_cut_score": round(score, 4),
        "scene_cut_count": len(posiciones),
        "scene_cut_max_diff": round(pico, 3),
        "scene_cut_positions": "|".join(str(p) for p in posiciones[:8]),
    }


def estimar_riesgo_corte(frames: np.ndarray) -> float:
    """Compatibilidad con la primera version del auditor."""

    return float(metricas_cortes_escena(frames)["scene_cut_score"])


def metricas_rostro_haar(video_path: str | Path, *, max_frames: int = 6) -> dict[str, Any]:
    """Mide caras visibles con cascadas Haar de OpenCV.

    Haar es una alerta liviana: ayuda a encontrar casos para revision, pero no prueba
    identidad ni hablante activo.
    """

    video_path = Path(video_path)
    frontal_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    profile_path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
    if not frontal_path.exists():
        return _metricas_rostro_no_disponible("haar_frontal_no_disponible")

    frontal = cv2.CascadeClassifier(str(frontal_path))
    profile = cv2.CascadeClassifier(str(profile_path)) if profile_path.exists() else None
    cap = cv2.VideoCapture(str(video_path))
    conteos: list[int] = []
    frontal_hits = 0
    profile_hits = 0
    centros = []
    areas = []
    boxes_por_frame = []
    frames_leidos = 0
    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for idx in indices_muestra(n_total, max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames_leidos += 1
            gris = _gris(frame)
            h, w = gris.shape[:2]
            cajas_frontales = detectar_cajas_haar_en_frame(gris, frontal, None)
            cajas_perfil = detectar_cajas_haar_en_frame(gris, None, profile)
            cajas = _filtrar_cajas(cajas_frontales + cajas_perfil, w, h)
            conteos.append(len(cajas))
            if cajas_frontales:
                frontal_hits += 1
            if cajas_perfil and not cajas_frontales:
                profile_hits += 1
            if cajas:
                dominante = max(cajas, key=lambda b: b[2] * b[3])
                x, y, bw, bh = dominante
                centros.append(((x + bw / 2) / w, (y + bh / 2) / h))
                areas.append((bw * bh) / (w * h))
            boxes_por_frame.append(
                {
                    "frame": int(idx),
                    "boxes": [_normalizar_caja(caja, w, h) for caja in cajas[:4]],
                }
            )
    finally:
        cap.release()

    if frames_leidos == 0:
        return _metricas_rostro_no_disponible("video_no_legible_para_rostros")

    frames_con_cara = sum(c > 0 for c in conteos)
    frames_multi = sum(c > 1 for c in conteos)
    ratio_cara = frames_con_cara / frames_leidos
    ratio_unica = sum(c == 1 for c in conteos) / frames_leidos
    ratio_multi = frames_multi / frames_leidos

    if centros:
        arr_centros = np.asarray(centros, dtype=np.float32)
        centro_promedio = arr_centros.mean(axis=0)
        distancias = np.sqrt(((arr_centros - centro_promedio) ** 2).sum(axis=1))
        dominant_face_shift = float(distancias.max(initial=0.0))
        arr_areas = np.asarray(areas, dtype=np.float32)
        face_box_area_jitter = float(arr_areas.std() / (arr_areas.mean() + 1e-6))
        track_stability_score = float(np.clip(1.0 - (dominant_face_shift * 2.0 + face_box_area_jitter * 0.7), 0.0, 1.0))
    else:
        dominant_face_shift = 1.0
        face_box_area_jitter = 1.0
        track_stability_score = 0.0

    pose_total = frontal_hits + profile_hits
    pose_available = pose_total > 0
    pose_score = frontal_hits / pose_total if pose_total else 0.5
    if not pose_available:
        pose_reason = "sin_cara_detectada_por_haar"
    elif profile_hits > frontal_hits:
        pose_reason = "perfil_proxy_haar"
    else:
        pose_reason = "frontal_proxy_haar"

    return {
        "face_detector": "haar",
        "frames_rostro_muestreados": frames_leidos,
        "face_count_score": round(float(ratio_unica), 4),
        "track_stability_score": round(track_stability_score, 4),
        "pose_score": round(float(pose_score), 4),
        "pose_available": pose_available,
        "pose_reason": pose_reason,
        "multi_face_risk": round(float(ratio_multi), 4),
        "ratio_cara": round(float(ratio_cara), 4),
        "avg_caras": round(float(np.mean(conteos)) if conteos else 0.0, 4),
        "max_caras": max(conteos) if conteos else 0,
        "dominant_face_shift": round(dominant_face_shift, 4),
        "face_box_area_jitter": round(face_box_area_jitter, 4),
        "face_boxes_summary": json.dumps(boxes_por_frame[:8], ensure_ascii=False, separators=(",", ":")),
        "face_notes": "haar_alerta_liviana",
    }


def detectar_cajas_haar_en_frame(
    gris: np.ndarray,
    frontal: cv2.CascadeClassifier | None = None,
    profile: cv2.CascadeClassifier | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detecta cajas con Haar en un frame gris."""

    h, w = gris.shape[:2]
    min_size = (max(32, int(min(h, w) * 0.10)), max(32, int(min(h, w) * 0.10)))
    cajas: list[tuple[int, int, int, int]] = []
    if frontal is not None:
        cajas.extend(frontal.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=6, minSize=min_size))
    if profile is not None:
        perfiles = list(profile.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=6, minSize=min_size))
        gris_flip = cv2.flip(gris, 1)
        perfiles_flip = profile.detectMultiScale(gris_flip, scaleFactor=1.1, minNeighbors=6, minSize=min_size)
        perfiles.extend([(w - x - bw, y, bw, bh) for (x, y, bw, bh) in perfiles_flip])
        cajas.extend(perfiles)
    return [(int(x), int(y), int(bw), int(bh)) for x, y, bw, bh in cajas]


def _normalizar_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim == 4:
        if frames.shape[-1] in (3, 4):
            frames = np.stack([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY) for f in frames], axis=0)
        elif frames.shape[1] in (3, 4):
            frames = np.stack([cv2.cvtColor(np.moveaxis(f, 0, -1).astype(np.uint8), cv2.COLOR_BGR2GRAY) for f in frames], axis=0)
        else:
            frames = frames[..., 0]
    if frames.ndim != 3:
        raise ValueError(f"Se esperaban frames [T,H,W], shape recibido: {frames.shape}")
    if frames.dtype.kind == "f" and frames.max(initial=0.0) <= 1.5:
        frames = frames * 255.0
    return np.clip(frames, 0, 255).astype(np.float32)


def _cargar_frames_npz(path: Path, *, max_frames: int, input_kind: str) -> VideoFrames:
    with np.load(path) as data:
        key = "rois" if "rois" in data.files else data.files[0]
        arr = data[key]
    n_total = int(arr.shape[0]) if arr.ndim >= 3 else 0
    idx = indices_muestra(n_total, max_frames)
    frames = _normalizar_frames(arr[idx] if idx else arr[:0])
    return VideoFrames(
        frames=frames,
        n_frames_total=n_total,
        fps=25.0,
        input_kind=input_kind,
        path=path,
        frames_esperados=len(idx),
    )


def _cargar_frames_video(path: Path, *, max_frames: int, input_kind: str) -> VideoFrames:
    cap = cv2.VideoCapture(str(path))
    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[np.ndarray] = []
        idx = indices_muestra(n_total, max_frames)
        if idx:
            for frame_idx in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(_gris(frame))
        else:
            leidos = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if leidos % 5 == 0:
                    frames.append(_gris(frame))
                leidos += 1
            n_total = leidos
            idx = list(range(len(frames)))
    finally:
        cap.release()

    arr = np.stack(frames, axis=0).astype(np.float32) if frames else np.zeros((0, 0, 0), dtype=np.float32)
    return VideoFrames(
        frames=arr,
        n_frames_total=n_total,
        fps=fps,
        input_kind=input_kind,
        path=path,
        frames_esperados=len(idx),
    )


def _gris(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _metricas_rostro_no_disponible(nota: str) -> dict[str, Any]:
    return {
        "face_detector": "no_disponible",
        "frames_rostro_muestreados": 0,
        "face_count_score": 1.0,
        "track_stability_score": 1.0,
        "pose_score": 0.5,
        "pose_available": False,
        "pose_reason": "not_available",
        "multi_face_risk": 0.0,
        "ratio_cara": "",
        "avg_caras": "",
        "max_caras": "",
        "dominant_face_shift": "",
        "face_box_area_jitter": "",
        "face_boxes_summary": "",
        "face_notes": nota,
    }


def _filtrar_cajas(cajas: list[Any], ancho: int, alto: int) -> list[tuple[int, int, int, int]]:
    candidatas: list[tuple[int, int, int, int]] = []
    area_total = max(1, ancho * alto)
    for caja in cajas:
        x, y, w, h = [int(v) for v in caja]
        area = w * h / area_total
        if area < 0.025:
            continue
        candidatas.append((x, y, w, h))
    candidatas.sort(key=lambda b: b[2] * b[3], reverse=True)

    filtradas: list[tuple[int, int, int, int]] = []
    for caja in candidatas:
        if all(_iou(caja, existente) < 0.35 for existente in filtradas):
            filtradas.append(caja)
    return filtradas


def _normalizar_caja(caja: tuple[int, int, int, int], ancho: int, alto: int) -> dict[str, float]:
    x, y, w, h = caja
    return {
        "x": round(x / max(ancho, 1), 4),
        "y": round(y / max(alto, 1), 4),
        "w": round(w / max(ancho, 1), 4),
        "h": round(h / max(alto, 1), 4),
    }


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0
