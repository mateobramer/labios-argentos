"""Audita candidatos con metadata, samples cortos y scores proxy.

Este script NO descarga videos completos. Para cada candidato intenta extraer samples
cortos con yt-dlp/ffmpeg, calcula evidencia visual/audio automatica y guarda un JSON
liviano en data_pipeline/discovery/outputs/sample_metadata/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from data_pipeline.discovery.src.common import (
    CANDIDATE_VIDEOS_CSV,
    SAMPLE_METADATA,
    SAMPLES,
    asegurar_directorios,
    clamp,
    cargar_json,
    comando_ffmpeg,
    comando_yt_dlp,
    configurar_salida_utf8,
    correr,
    guardar_json,
    leer_csv,
    leer_jsonl,
    score_intervalo,
    score_rango,
    slug,
    texto_normalizado,
    to_float,
)
from data_pipeline.discovery.src.search_youtube_candidates import inferir_acento, inferir_source_type, inferir_speaker_count


ARG_HINTS = [
    "argentina",
    "argentino",
    "buenos aires",
    "rioplatense",
    "caba",
    "che",
    "vos",
    "voseo",
    "coscu",
    "migue granados",
    "olga",
    "luzu",
    "urbana play",
    "gelatina",
]

_MP_LANDMARKER: Any | None = None
_MP_MODULE: Any | None = None
_MP_STATUS: str | None = None
_ASR_MODELS: dict[str, Any] = {}


def obtener_metadata(url: str, timeout: int = 120) -> tuple[dict[str, Any], str]:
    cmd = comando_yt_dlp() + ["--dump-json", "--skip-download", "--no-playlist", url]
    result = correr(cmd, timeout=timeout)
    if result.returncode != 0:
        return {}, (result.stderr or result.stdout).strip()[:500]
    rows = leer_jsonl(result.stdout)
    return (rows[0] if rows else {}), ""


def timestamps_samples(duration: float, sample_count: int, sample_seconds: int) -> list[dict[str, float]]:
    if duration <= 0:
        return []
    margen_inicio = min(90.0, max(15.0, duration * 0.08))
    margen_fin = min(90.0, max(15.0, duration * 0.08))
    disponible = max(sample_seconds, duration - margen_inicio - margen_fin)
    if disponible <= sample_seconds:
        starts = [max(0.0, (duration - sample_seconds) / 2)]
    else:
        n = max(1, sample_count)
        if n == 1:
            starts = [margen_inicio + disponible * 0.5 - sample_seconds / 2]
        else:
            starts = [
                margen_inicio + (disponible - sample_seconds) * (i + 0.5) / n
                for i in range(n)
            ]
    return [
        {"index": idx + 1, "start": round(max(0.0, s), 2), "end": round(min(duration, s + sample_seconds), 2)}
        for idx, s in enumerate(starts)
    ]


def descargar_sample(url: str, video_id: str, sample: dict[str, float], reuse: bool = True) -> tuple[Path | None, str]:
    out_dir = SAMPLES / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{int(sample['index']):02d}"
    existentes = sorted(out_dir.glob(stem + ".*"))
    existentes = [p for p in existentes if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if reuse and existentes:
        return existentes[0], "reused"

    template = str(out_dir / (stem + ".%(ext)s"))
    seccion = f"*{sample['start']}-{sample['end']}"
    ffmpeg_location = str(Path(comando_ffmpeg()))
    cmd = comando_yt_dlp() + [
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--ffmpeg-location",
        ffmpeg_location,
        "--force-keyframes-at-cuts",
        "--download-sections",
        seccion,
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "-o",
        template,
        url,
    ]
    result = correr(cmd, timeout=240)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()[:700]
    existentes = sorted(out_dir.glob(stem + ".*"))
    existentes = [p for p in existentes if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not existentes:
        return None, "sample_no_generado"
    return existentes[0], "downloaded"


def metricas_visuales(sample_path: Path) -> dict[str, Any]:
    try:
        from cleaning.visual_quality.src.visual_quality_metrics import cargar_frames, metricas_calidad_frames, metricas_rostro_haar
    except Exception as exc:  # pragma: no cover - depende del entorno local
        return {"visual_error": f"blocked_missing_visual_dependency:{exc}"}

    try:
        video = cargar_frames(sample_path, max_frames=60, input_kind="raw_clip")
        calidad = metricas_calidad_frames(video)
        rostro = metricas_rostro_haar(sample_path, max_frames=8)
    except Exception as exc:
        return {"visual_error": f"visual_metrics_error:{exc}"}

    face_size_ratio = extraer_face_size_ratio(rostro)
    mouth_visible_ratio = to_float(calidad.get("mouth_visibility_score"))
    single_speaker = (
        to_float(rostro.get("face_count_score"), 0.6)
        * to_float(rostro.get("track_stability_score"), 0.6)
        * (1.0 - to_float(rostro.get("multi_face_risk"), 0.0))
    )
    metrics = {
        "face_detect_rate": to_float(rostro.get("ratio_cara"), 0.0),
        "mouth_detect_rate": mouth_visible_ratio,
        "mouth_size_ratio": round(face_size_ratio * 0.20, 4),
        "face_size_ratio": round(face_size_ratio, 4),
        "center_stability": to_float(rostro.get("track_stability_score"), 0.0),
        "frontal_proxy": to_float(rostro.get("pose_score"), 0.5),
        "side_profile_proxy": round(1.0 - to_float(rostro.get("pose_score"), 0.5), 4),
        "occlusion_proxy": round(1.0 - mouth_visible_ratio, 4),
        "black_frame_rate": to_float(calidad.get("frac_oscuros"), 0.0),
        "scene_cut_proxy": to_float(calidad.get("scene_cut_score"), 0.0),
        "multi_face_rate": to_float(rostro.get("multi_face_risk"), 0.0),
        "mouth_visible_ratio": mouth_visible_ratio,
        "single_speaker_visual_proxy": round(clamp(single_speaker), 4),
        "brightness_score": to_float(calidad.get("brightness_score"), 0.0),
        "contrast_score": to_float(calidad.get("contrast_score"), 0.0),
        "blur_score": to_float(calidad.get("blur_score"), 0.0),
        "mouth_activity_score": to_float(calidad.get("mouth_activity_score"), 0.0),
        "mouth_texture_score": to_float(calidad.get("mouth_texture_score"), 0.0),
        "face_detector": rostro.get("face_detector", ""),
        "face_notes": rostro.get("face_notes", ""),
        "visual_backend": "opencv_haar_proxy",
    }
    mp_metrics = metricas_mediapipe(sample_path)
    if mp_metrics.get("mediapipe_status") == "ok":
        metrics.update(fusionar_mediapipe(metrics, mp_metrics))
    else:
        metrics.update(mp_metrics)
    metrics["visual_quality_score"] = round(calcular_visual_score(metrics), 2)
    decision, reasons = decidir_visual(metrics)
    metrics["visual_decision"] = decision
    metrics["visual_reasons"] = reasons
    return metrics


def metricas_mediapipe(sample_path: Path) -> dict[str, Any]:
    global _MP_LANDMARKER, _MP_MODULE, _MP_STATUS
    if _MP_STATUS == "blocked":
        return {"mediapipe_status": "blocked_unavailable"}
    try:
        from preprocessing.src.auditar_calidad_visual import analizar_clip, crear_landmarker
    except Exception as exc:
        _MP_STATUS = "blocked"
        return {"mediapipe_status": f"blocked_missing_mediapipe:{exc}"}
    try:
        if _MP_LANDMARKER is None or _MP_MODULE is None:
            _MP_LANDMARKER, _MP_MODULE = crear_landmarker(num_faces=3)
        data = analizar_clip(sample_path, n_frames=5, landmarker=_MP_LANDMARKER, mp=_MP_MODULE)
    except Exception as exc:
        return {"mediapipe_status": f"mediapipe_error:{exc}"}
    return {
        "mediapipe_status": "ok",
        "mediapipe_face_detect_rate": to_float(data.get("ratio_cara")),
        "mediapipe_multi_face_rate": to_float(data.get("ratio_multi_cara")),
        "mediapipe_mouth_visible_ratio": to_float(data.get("ratio_boca_visible")),
        "mediapipe_mouth_centered_ratio": to_float(data.get("ratio_boca_centrada")),
        "mediapipe_max_faces": data.get("max_caras", ""),
        "mediapipe_visual_reasons": data.get("visual_reasons", ""),
    }


def fusionar_mediapipe(base: dict[str, Any], mp_metrics: dict[str, Any]) -> dict[str, Any]:
    face = to_float(mp_metrics.get("mediapipe_face_detect_rate"))
    mouth = to_float(mp_metrics.get("mediapipe_mouth_visible_ratio"))
    centered = to_float(mp_metrics.get("mediapipe_mouth_centered_ratio"))
    multi = to_float(mp_metrics.get("mediapipe_multi_face_rate"))
    single = face * (1.0 - multi) * max(0.5, centered)
    out = dict(mp_metrics)
    out.update(
        {
            "face_detect_rate": max(to_float(base.get("face_detect_rate")), face),
            "mouth_detect_rate": max(to_float(base.get("mouth_detect_rate")), mouth),
            "mouth_visible_ratio": max(to_float(base.get("mouth_visible_ratio")), mouth),
            "center_stability": max(to_float(base.get("center_stability")), centered),
            "multi_face_rate": multi,
            "single_speaker_visual_proxy": max(to_float(base.get("single_speaker_visual_proxy")), clamp(single)),
            "visual_backend": "mediapipe_face_landmarker+opencv_proxy",
            "face_detector": "mediapipe+haar",
        }
    )
    if mouth >= 0.8:
        out["occlusion_proxy"] = min(to_float(base.get("occlusion_proxy")), 1.0 - mouth)
    return out


def extraer_face_size_ratio(rostro: dict[str, Any]) -> float:
    raw = rostro.get("face_boxes_summary") or ""
    try:
        frames = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return 0.0
    areas = []
    for frame in frames:
        boxes = frame.get("boxes") or []
        if boxes:
            box = boxes[0]
            areas.append(to_float(box.get("w")) * to_float(box.get("h")))
    return float(mean(areas)) if areas else 0.0


def calcular_visual_score(m: dict[str, Any]) -> float:
    face_size_score = score_intervalo(to_float(m.get("face_size_ratio")), 0.015, 0.05, 0.30, 0.55)
    return 100.0 * (
        0.18 * clamp(to_float(m.get("face_detect_rate")))
        + 0.18 * clamp(to_float(m.get("mouth_visible_ratio")))
        + 0.12 * face_size_score
        + 0.12 * clamp(to_float(m.get("single_speaker_visual_proxy")))
        + 0.10 * clamp(to_float(m.get("center_stability")))
        + 0.08 * clamp(to_float(m.get("frontal_proxy")))
        + 0.08 * (1.0 - clamp(to_float(m.get("scene_cut_proxy"))))
        + 0.06 * (1.0 - clamp(to_float(m.get("black_frame_rate"))))
        + 0.04 * clamp(to_float(m.get("blur_score")))
        + 0.04 * clamp(to_float(m.get("contrast_score")))
    )


def decidir_visual(m: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if to_float(m.get("face_detect_rate")) < 0.35:
        reasons.append("face_detect_rate_bajo")
    if to_float(m.get("mouth_visible_ratio")) < 0.35:
        reasons.append("mouth_visible_ratio_bajo")
    if to_float(m.get("face_size_ratio")) and to_float(m.get("face_size_ratio")) < 0.025:
        reasons.append("cara_demasiado_chica")
    if to_float(m.get("multi_face_rate")) > 0.55:
        reasons.append("multi_face_rate_alto")
    if to_float(m.get("scene_cut_proxy")) > 0.75:
        reasons.append("scene_cut_proxy_alto")
    if to_float(m.get("black_frame_rate")) > 0.2:
        reasons.append("black_frame_rate_alto")
    if to_float(m.get("frontal_proxy")) < 0.25:
        reasons.append("perfil_extremo_proxy")

    score = to_float(m.get("visual_quality_score"))
    if reasons and any(r in reasons for r in ["face_detect_rate_bajo", "mouth_visible_ratio_bajo", "cara_demasiado_chica"]):
        return "reject", reasons
    if score >= 88:
        return "strong_accept", reasons or ["visual_proxy_alto"]
    if score >= 80:
        return "accept", reasons or ["visual_proxy_suficiente"]
    if score >= 62:
        return "maybe_review", reasons or ["visual_proxy_intermedio"]
    return "reject", reasons or ["visual_score_bajo"]


def duracion_video(path: Path) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if frames > 0 and fps > 0:
            return frames / fps
    except Exception:
        pass
    return 0.0


def metricas_audio(sample_path: Path, *, run_asr: bool = False, asr_model: str = "small") -> dict[str, Any]:
    ffmpeg = comando_ffmpeg()
    duration = duracion_video(sample_path)
    silence_cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(sample_path),
        "-af",
        "silencedetect=n=-35dB:d=0.35",
        "-f",
        "null",
        "-",
    ]
    silence = correr(silence_cmd, timeout=120)
    stderr = silence.stderr or silence.stdout
    silence_seconds = estimar_silencio(stderr)
    speech_presence = 0.0 if duration <= 0 else clamp(1.0 - silence_seconds / duration)

    vol_cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(sample_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    vol = correr(vol_cmd, timeout=120)
    volume_text = vol.stderr or vol.stdout
    mean_volume = extraer_db(volume_text, "mean_volume")
    max_volume = extraer_db(volume_text, "max_volume")
    volume_score = score_intervalo(mean_volume, -55.0, -34.0, -12.0, -3.0) if mean_volume is not None else 0.45
    clipping_risk = 1.0 if max_volume is not None and max_volume > -1.0 else 0.0
    audio_quality_score = 100.0 * (
        0.58 * speech_presence
        + 0.27 * volume_score
        + 0.10 * (1.0 - clipping_risk)
        + 0.05 * (1.0 if duration >= 8 else 0.5)
    )
    metrics = {
        "duration_seconds": round(duration, 3),
        "speech_presence_ratio": round(speech_presence, 4),
        "asr_text_length": "",
        "language_detected": "",
        "likely_spanish": "",
        "no_speech_probability": "",
        "audio_quality_proxy": round(volume_score, 4),
        "background_music_proxy": "",
        "noise_proxy": "",
        "speech_density": round(speech_presence, 4),
        "likely_rioplatense_hint": "",
        "mean_volume_db": round(mean_volume, 3) if mean_volume is not None else "",
        "max_volume_db": round(max_volume, 3) if max_volume is not None else "",
        "audio_quality_score": round(audio_quality_score, 2),
        "audio_backend": "ffmpeg_silence_volumedetect",
        "asr_status": estado_asr_disponible(),
    }
    if run_asr:
        metrics.update(metricas_asr(sample_path, asr_model=asr_model, duration=duration))
    decision, reasons = decidir_audio(metrics)
    metrics["audio_decision"] = decision
    metrics["audio_reasons"] = reasons
    return metrics


def estado_asr_disponible() -> str:
    if importlib.util.find_spec("faster_whisper") is not None:
        return "not_run_faster_whisper_available"
    if importlib.util.find_spec("whisper") is not None:
        return "not_run_whisper_available"
    return "blocked_missing_asr_dependency"


def metricas_asr(sample_path: Path, *, asr_model: str, duration: float) -> dict[str, Any]:
    if importlib.util.find_spec("faster_whisper") is None:
        return {"asr_status": "blocked_missing_asr_dependency"}
    try:
        from faster_whisper import WhisperModel

        if asr_model not in _ASR_MODELS:
            _ASR_MODELS[asr_model] = WhisperModel(asr_model, device="cpu", compute_type="int8")
        model = _ASR_MODELS[asr_model]
        segments_iter, info = model.transcribe(str(sample_path), language="es", vad_filter=True)
        segments = list(segments_iter)
    except Exception as exc:
        return {"asr_status": f"asr_error:{exc}"}

    text = " ".join(getattr(seg, "text", "").strip() for seg in segments).strip()
    speech_seconds = sum(max(0.0, float(getattr(seg, "end", 0.0)) - float(getattr(seg, "start", 0.0))) for seg in segments)
    no_speech_vals = [getattr(seg, "no_speech_prob", None) for seg in segments]
    no_speech_vals = [float(v) for v in no_speech_vals if v is not None]
    language = getattr(info, "language", "") or ""
    language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    likely_spanish = language == "es" or language_probability >= 0.60
    return {
        "asr_status": f"ok_faster_whisper_{asr_model}",
        "asr_text_length": len(text),
        "language_detected": language,
        "likely_spanish": str(likely_spanish),
        "language_probability": round(language_probability, 4),
        "no_speech_probability": round(sum(no_speech_vals) / len(no_speech_vals), 4) if no_speech_vals else "",
        "speech_density": round(clamp(speech_seconds / duration), 4) if duration > 0 else "",
        "asr_preview": text[:180],
    }


def estimar_silencio(text: str) -> float:
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    total = 0.0
    for start, end in zip(starts, ends):
        total += max(0.0, end - start)
    return total


def extraer_db(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}:\s*(-?[0-9.]+)\s*dB", text)
    return float(match.group(1)) if match else None


def decidir_audio(m: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if to_float(m.get("speech_presence_ratio")) < 0.35:
        reasons.append("habla_insuficiente_proxy")
    if to_float(m.get("audio_quality_proxy")) < 0.25:
        reasons.append("audio_muy_bajo_o_saturado_proxy")
    score = to_float(m.get("audio_quality_score"))
    if reasons:
        return "reject", reasons
    if score >= 86:
        return "strong_accept", ["audio_proxy_alto"]
    if score >= 75:
        return "accept", ["audio_proxy_suficiente"]
    if score >= 58:
        return "maybe_review", ["audio_proxy_intermedio"]
    return "reject", ["audio_score_bajo"]


def score_contexto(info: dict[str, Any], candidate: dict[str, str]) -> dict[str, Any]:
    title = str(info.get("title") or candidate.get("title") or "")
    channel = str(info.get("channel") or info.get("uploader") or candidate.get("channel") or "")
    description = str(info.get("description") or "")
    tags = info.get("tags") or []
    duration = to_float(info.get("duration") or candidate.get("duration")) / 60.0
    source_type = inferir_source_type(title, channel, description)
    accent = candidate.get("expected_accent") or inferir_acento(title, channel, description, tags)
    speaker_count = candidate.get("expected_speaker_count") or inferir_speaker_count(source_type, title)
    texto = texto_normalizado(title, channel, description, " ".join(tags))
    positives = []
    negatives = []
    score = 45.0
    if any(h in texto for h in ARG_HINTS) or "argentino" in accent:
        score += 25.0
        positives.append("fuente_argentina_probable")
    else:
        negatives.append("acento_fuente_unknown")
    if duration >= 20:
        score += 10.0
        positives.append("formato_largo")
    if source_type in {"podcast", "interview", "streamer", "monologue", "educational", "conversation", "vlog"}:
        score += 10.0
        positives.append(f"formato_{source_type}")
    if speaker_count in {"1", "1-2"}:
        score += 8.0
        positives.append("speaker_count_usable")
    if any(k in texto for k in ["musica", "cancion", "compilado", "shorts", "tiktok", "doblaje"]):
        score -= 35.0
        negatives.append("formato_a_evitar_keyword")
    if any(k in texto for k in ["panel", "mesa", "debate"]):
        score -= 10.0
        negatives.append("muchos_hablantes_posible")
    score = max(0.0, min(100.0, score))
    return {
        "context_score": round(score, 2),
        "context_reasons": positives + negatives,
        "source_type": source_type,
        "expected_accent": accent,
        "expected_speaker_count": speaker_count,
    }


def agregar_promedios(sample_results: list[dict[str, Any]]) -> dict[str, Any]:
    visual_keys = [
        "face_detect_rate",
        "mouth_detect_rate",
        "mouth_size_ratio",
        "face_size_ratio",
        "center_stability",
        "frontal_proxy",
        "side_profile_proxy",
        "occlusion_proxy",
        "black_frame_rate",
        "scene_cut_proxy",
        "multi_face_rate",
        "mouth_visible_ratio",
        "single_speaker_visual_proxy",
        "visual_quality_score",
    ]
    audio_keys = [
        "speech_presence_ratio",
        "audio_quality_proxy",
        "speech_density",
        "audio_quality_score",
    ]
    out: dict[str, Any] = {}
    for key in visual_keys + audio_keys:
        vals = [to_float(s.get(key), None) for s in sample_results if s.get(key) not in ("", None)]
        vals = [v for v in vals if v is not None]
        out[key] = round(mean(vals), 4) if vals else ""
    out["samples_ok"] = sum(1 for s in sample_results if s.get("sample_path"))
    out["samples_total"] = len(sample_results)

    visual_decisions = [str(s.get("visual_decision") or "") for s in sample_results]
    audio_decisions = [str(s.get("audio_decision") or "") for s in sample_results]
    out["visual_decision"] = combinar_decisiones(visual_decisions, to_float(out.get("visual_quality_score")))
    out["audio_decision"] = combinar_decisiones(audio_decisions, to_float(out.get("audio_quality_score")))
    out["visual_reasons"] = sorted({r for s in sample_results for r in s.get("visual_reasons", [])})
    out["audio_reasons"] = sorted({r for s in sample_results for r in s.get("audio_reasons", [])})
    return out


def combinar_decisiones(decisions: list[str], score: float) -> str:
    if not decisions:
        return "reject"
    rejects = decisions.count("reject")
    if rejects >= max(1, len(decisions) // 2 + 1):
        return "reject"
    if score >= 88:
        return "strong_accept"
    if score >= 80:
        return "accept"
    if score >= 62:
        return "maybe_review"
    return "reject"


def auditar(
    candidate: dict[str, str],
    sample_count: int,
    sample_seconds: int,
    dry_run: bool,
    reuse_samples: bool,
    run_asr: bool = False,
    asr_model: str = "small",
) -> dict[str, Any]:
    url = candidate.get("url", "")
    metadata, error = obtener_metadata(url)
    video_id = str(metadata.get("id") or candidate.get("video_id") or slug(url))
    duration = to_float(metadata.get("duration") or candidate.get("duration"))
    audit: dict[str, Any] = {
        "url": url,
        "video_id": video_id,
        "title": metadata.get("title") or candidate.get("title") or "",
        "channel": metadata.get("channel") or metadata.get("uploader") or candidate.get("channel") or "",
        "duration": duration,
        "duration_minutes": round(duration / 60.0, 3) if duration else "",
        "width": metadata.get("width") or candidate.get("width") or "",
        "height": metadata.get("height") or candidate.get("height") or "",
        "fps": metadata.get("fps") or candidate.get("fps") or "",
        "upload_date": metadata.get("upload_date") or candidate.get("upload_date") or "",
        "view_count": metadata.get("view_count") or candidate.get("view_count") or "",
        "language": metadata.get("language") or candidate.get("language") or "",
        "tags": metadata.get("tags") or [],
        "metadata_error": error,
        "samples": [],
    }
    audit.update(score_contexto(metadata, candidate))
    if error:
        audit["audit_status"] = "metadata_error"
        return audit

    audit_previo = cargar_auditoria_previa(video_id)
    samples_previos = {
        int(s.get("index")): s
        for s in audit_previo.get("samples", [])
        if str(s.get("index", "")).isdigit()
    }
    for sample in timestamps_samples(duration, sample_count, sample_seconds):
        sample_row: dict[str, Any] = dict(sample)
        if dry_run:
            sample_row["sample_status"] = "dry_run"
            audit["samples"].append(sample_row)
            continue
        sample_path, status = descargar_sample(url, video_id, sample, reuse=reuse_samples)
        sample_row["sample_status"] = status
        sample_row["sample_path"] = str(sample_path) if sample_path else ""
        if sample_path:
            previo = samples_previos.get(int(sample["index"]))
            if previo and previo.get("visual_quality_score") not in ("", None):
                sample_row.update(metricas_visuales_cacheadas(previo))
            else:
                sample_row.update(metricas_visuales(sample_path))
            sample_row.update(metricas_audio(sample_path, run_asr=run_asr, asr_model=asr_model))
        audit["samples"].append(sample_row)

    audit.update(agregar_promedios(audit["samples"]))
    audit["audit_status"] = "ok" if audit.get("samples_ok", 0) else ("dry_run" if dry_run else "sample_error")
    audit["uncertainty"] = calcular_uncertainty(audit)
    return audit


def cargar_auditoria_previa(video_id: str) -> dict[str, Any]:
    path = SAMPLE_METADATA / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        return cargar_json(path)
    except Exception:
        return {}


def metricas_visuales_cacheadas(sample: dict[str, Any]) -> dict[str, Any]:
    claves = [
        "face_detect_rate",
        "mouth_detect_rate",
        "mouth_size_ratio",
        "face_size_ratio",
        "center_stability",
        "frontal_proxy",
        "side_profile_proxy",
        "occlusion_proxy",
        "black_frame_rate",
        "scene_cut_proxy",
        "multi_face_rate",
        "mouth_visible_ratio",
        "single_speaker_visual_proxy",
        "brightness_score",
        "contrast_score",
        "blur_score",
        "mouth_activity_score",
        "mouth_texture_score",
        "face_detector",
        "face_notes",
        "visual_backend",
        "mediapipe_status",
        "mediapipe_face_detect_rate",
        "mediapipe_multi_face_rate",
        "mediapipe_mouth_visible_ratio",
        "mediapipe_mouth_centered_ratio",
        "mediapipe_max_faces",
        "mediapipe_visual_reasons",
        "visual_quality_score",
        "visual_decision",
        "visual_reasons",
    ]
    return {k: sample[k] for k in claves if k in sample}


def calcular_uncertainty(audit: dict[str, Any]) -> str:
    reasons = []
    if audit.get("samples_ok", 0) < max(1, audit.get("samples_total", 0)):
        reasons.append("samples_incompletos")
    if audit.get("expected_accent") == "unknown":
        reasons.append("acento_unknown")
    if not audit.get("language"):
        reasons.append("language_metadata_missing")
    if audit.get("audio_decision") in {"maybe_review", "reject"}:
        reasons.append("audio_proxy_debil")
    if not reasons:
        return "low"
    if len(reasons) <= 2:
        return "medium:" + "|".join(reasons)
    return "high:" + "|".join(reasons)


def main() -> int:
    configurar_salida_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CANDIDATE_VIDEOS_CSV)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-seconds", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true", help="Solo metadata y timestamps; no baja samples.")
    parser.add_argument("--no-reuse-samples", action="store_true")
    parser.add_argument("--run-asr", action="store_true", help="Corre ASR sobre samples con faster-whisper si esta disponible.")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "base", "small", "medium"], help="Modelo ASR para samples.")
    args = parser.parse_args()

    asegurar_directorios()
    rows = [r for r in leer_csv(args.input) if r.get("url") and not str(r.get("notes", "")).startswith("prefilter_reject")]
    selected = rows[args.offset : args.offset + args.limit]
    for idx, row in enumerate(selected, start=1):
        audit = auditar(
            row,
            args.sample_count,
            args.sample_seconds,
            args.dry_run,
            not args.no_reuse_samples,
            run_asr=args.run_asr,
            asr_model=args.asr_model,
        )
        video_id = audit.get("video_id") or slug(row.get("url", ""))
        path = SAMPLE_METADATA / f"{video_id}.json"
        guardar_json(path, audit)
        print(f"[{idx}/{len(selected)}] {audit.get('audit_status')} {audit.get('title', '')[:80]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
