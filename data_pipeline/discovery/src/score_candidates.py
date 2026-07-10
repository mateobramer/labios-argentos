"""Consolida auditorias y genera scores, shortlist y reportes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from data_pipeline.discovery.src.common import (
    CANDIDATE_SCORES_CSV,
    CANDIDATE_VIDEOS_CSV,
    DIVERSITY_CSV,
    DIVERSITY_FIELDS,
    INGEST_PLAN_MD,
    REJECTED_CSV,
    REJECTED_FIELDS,
    REVIEW_INDEX_MD,
    SAMPLE_METADATA,
    SCORE_FIELDS,
    SHORTLIST_CSV,
    SHORTLIST_FIELDS,
    TARGET_PROGRESS_MD,
    asegurar_directorios,
    cargar_json,
    configurar_salida_utf8,
    decision_orden,
    escribir_csv,
    formatear_minutos,
    leer_csv,
    to_float,
)
from data_pipeline.discovery.src.estimate_usable_clips import DEFAULT_CLIPS_PER_MINUTE, estimar_clips_aceptados, estimar_minutos_utiles


TARGET_NEW_ACCEPTED_CLIPS = 12000
STRETCH_NEW_ACCEPTED_CLIPS = 20000
TARGET_USABLE_MINUTES_LOW = 600
TARGET_USABLE_MINUTES_HIGH = 900
MAX_CLIPS_PER_SOURCE = 1800
MAX_MINUTES_PER_SOURCE = 60


def cargar_auditorias(path: Path = SAMPLE_METADATA) -> dict[str, dict[str, Any]]:
    audits = {}
    if not path.exists():
        return audits
    for json_path in sorted(path.glob("*.json")):
        data = cargar_json(json_path)
        key = str(data.get("video_id") or json_path.stem)
        audits[key] = data
    return audits


def row_desde_candidate(candidate: dict[str, str], audit: dict[str, Any] | None, clips_per_minute: float) -> dict[str, Any]:
    audit = audit or {}
    visual_score = to_float(audit.get("visual_quality_score"), 0.0)
    audio_score = to_float(audit.get("audio_quality_score"), 0.0)
    asr_eval = evaluar_asr(audit.get("samples", []))
    if asr_eval["max_audio_score"] is not None:
        audio_score = min(audio_score, asr_eval["max_audio_score"])
    context_score = to_float(audit.get("context_score"), 0.0)
    total = round(0.50 * visual_score + 0.30 * audio_score + 0.20 * context_score, 2)
    visual_decision = audit.get("visual_decision") or ("reject" if not audit else "maybe_review")
    audio_decision = audit.get("audio_decision") or ("reject" if not audit else "maybe_review")
    reasons = razones(audit, candidate, total, visual_score, audio_score, context_score)
    reasons.extend(asr_eval["reasons"])
    decision = decidir_final(
        total_score=total,
        visual_score=visual_score,
        audio_score=audio_score,
        context_score=context_score,
        expected_accent=str(audit.get("expected_accent") or candidate.get("expected_accent") or "unknown"),
        visual_decision=str(visual_decision),
        audio_decision=str(audio_decision),
        audit_status=str(audit.get("audit_status") or "missing_audit"),
    )
    recommended_use = recomendar_uso(decision, to_float(audit.get("duration_minutes") or candidate.get("duration_minutes")))
    usable = estimar_minutos_utiles(
        video_duration_minutes=to_float(audit.get("duration_minutes") or candidate.get("duration_minutes")),
        speech_presence_ratio=to_float(audit.get("speech_presence_ratio"), 0.55),
        mouth_visible_ratio=to_float(audit.get("mouth_visible_ratio"), visual_score / 100 if visual_score else 0.45),
        single_speaker_ratio=to_float(audit.get("single_speaker_visual_proxy"), 0.65),
        visual_accept_ratio=visual_score / 100 if visual_score else 0.0,
    )
    clips = estimar_clips_aceptados(usable, clips_per_minute)
    if decision in {"strong_accept", "accept"} and (usable <= 0 or clips <= 0):
        reasons.append("sin_minutos_utiles_estimados")
        decision = "maybe_review"
        recommended_use = "manual_review"
    if decision not in {"strong_accept", "accept"}:
        clips = 0.0 if decision == "reject" else clips * 0.35
        usable = 0.0 if decision == "reject" else usable * 0.35
    row = {
        "url": audit.get("url") or candidate.get("url") or "",
        "title": audit.get("title") or candidate.get("title") or "",
        "channel": audit.get("channel") or candidate.get("channel") or "",
        "video_id": audit.get("video_id") or candidate.get("video_id") or "",
        "duration_minutes": round(to_float(audit.get("duration_minutes") or candidate.get("duration_minutes")), 3),
        "width": audit.get("width") or candidate.get("width") or "",
        "height": audit.get("height") or candidate.get("height") or "",
        "fps": audit.get("fps") or candidate.get("fps") or "",
        "source_type": audit.get("source_type") or candidate.get("source_type") or "other",
        "expected_accent": audit.get("expected_accent") or candidate.get("expected_accent") or "unknown",
        "visual_quality_score": round(visual_score, 2),
        "audio_quality_score": round(audio_score, 2),
        "context_score": round(context_score, 2),
        "total_score": total,
        "visual_decision": visual_decision,
        "audio_decision": audio_decision,
        "decision": decision,
        "reasons": "|".join(reasons),
        "recommended_use": recommended_use,
        "usable_minutes_estimate": round(usable, 2),
        "accepted_clips_estimate": round(clips, 0),
        "uncertainty": ajustar_uncertainty(audit.get("uncertainty") or "high:missing_audit", asr_eval),
    }
    return row


def evaluar_asr(samples: list[dict[str, Any]]) -> dict[str, Any]:
    asr_samples = [s for s in samples if str(s.get("asr_status", "")).startswith("ok_")]
    if not asr_samples:
        statuses = {str(s.get("asr_status", "")) for s in samples if s.get("asr_status")}
        if any("blocked_missing_asr_dependency" in s for s in statuses):
            return {"max_audio_score": None, "reasons": ["asr_blocked_missing_dependency"], "status": "blocked"}
        return {"max_audio_score": None, "reasons": ["asr_not_run"], "status": "not_run"}

    likely_spanish = [
        str(s.get("likely_spanish", "")).lower() == "true" or str(s.get("language_detected", "")).lower() == "es"
        for s in asr_samples
    ]
    spanish_ratio = sum(likely_spanish) / len(likely_spanish)
    avg_text_len = mean([to_float(s.get("asr_text_length")) for s in asr_samples])
    densities = [to_float(s.get("speech_density"), 0.0) for s in asr_samples if s.get("speech_density") not in ("", None)]
    avg_density = mean(densities) if densities else 0.0
    no_speech = [to_float(s.get("no_speech_probability"), 0.0) for s in asr_samples if s.get("no_speech_probability") not in ("", None)]
    avg_no_speech = mean(no_speech) if no_speech else 0.0

    reasons: list[str] = []
    max_audio_score: float | None = None
    if spanish_ratio < 0.67:
        reasons.append("asr_no_confirma_espanol")
        max_audio_score = 55.0
    if avg_text_len < 60:
        reasons.append("asr_texto_muy_corto")
        max_audio_score = min(max_audio_score or 100.0, 60.0)
    if avg_density < 0.35:
        reasons.append("asr_baja_densidad_habla")
        max_audio_score = min(max_audio_score or 100.0, 65.0)
    if avg_no_speech > 0.60:
        reasons.append("asr_no_speech_alto")
        max_audio_score = min(max_audio_score or 100.0, 65.0)
    if not reasons:
        reasons.append("asr_espanol_y_habla_confirmados")
    return {"max_audio_score": max_audio_score, "reasons": reasons, "status": "ok"}


def ajustar_uncertainty(uncertainty: str, asr_eval: dict[str, Any]) -> str:
    if asr_eval["status"] == "ok" and "asr_espanol_y_habla_confirmados" in asr_eval["reasons"]:
        return uncertainty.replace("audio_proxy_debil", "audio_asr_confirmado")
    if asr_eval["status"] == "ok" and asr_eval["max_audio_score"] is not None:
        return "high:asr_quality_penalty"
    return uncertainty


def razones(
    audit: dict[str, Any],
    candidate: dict[str, str],
    total: float,
    visual_score: float,
    audio_score: float,
    context_score: float,
) -> list[str]:
    out: list[str] = []
    out.extend(str(r) for r in audit.get("visual_reasons", []) if r)
    out.extend(str(r) for r in audit.get("audio_reasons", []) if r)
    out.extend(str(r) for r in audit.get("context_reasons", []) if r)
    if not audit:
        out.append("missing_audit_samples")
    if total >= 85:
        out.append("total_score_alto")
    if visual_score >= 80:
        out.append("visual_score_sobre_threshold")
    if audio_score >= 75:
        out.append("audio_score_sobre_threshold")
    if context_score < 70:
        out.append("contexto_incierto")
    if candidate.get("notes", "").startswith("prefilter_reject"):
        out.append(candidate["notes"])
    return sorted(set(out))


def decidir_final(
    *,
    total_score: float,
    visual_score: float,
    audio_score: float,
    context_score: float,
    expected_accent: str,
    visual_decision: str,
    audio_decision: str,
    audit_status: str,
) -> str:
    if audit_status not in {"ok", "dry_run"}:
        return "reject"
    if visual_decision == "reject" or audio_decision == "reject":
        return "reject"
    accent_ok = "argentino" in expected_accent or "rioplatense" in expected_accent
    if total_score >= 90 and visual_score >= 86 and audio_score >= 82 and context_score >= 80 and accent_ok:
        return "strong_accept"
    if total_score >= 85 and visual_score >= 80 and audio_score >= 75 and accent_ok:
        return "accept"
    if total_score >= 72 and visual_score >= 62 and audio_score >= 58:
        return "maybe_review"
    return "reject"


def recomendar_uso(decision: str, duration_minutes: float) -> str:
    if decision == "strong_accept":
        return "ingest_full" if duration_minutes <= 90 else "ingest_partial"
    if decision == "accept":
        return "ingest_full" if duration_minutes <= 60 else "ingest_partial"
    if decision == "maybe_review":
        return "manual_review"
    return "reject"


def aplicar_caps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    por_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["decision"] in {"strong_accept", "accept"}:
            por_channel[str(row["channel"])].append(row)

    for channel_rows in por_channel.values():
        channel_rows.sort(
            key=lambda r: (
                -to_float(r["total_score"]),
                -to_float(r["visual_quality_score"]),
                -to_float(r["usable_minutes_estimate"]),
            )
        )
        clips_acc = 0.0
        minutes_acc = 0.0
        for row in channel_rows:
            next_clips = clips_acc + to_float(row["accepted_clips_estimate"])
            next_minutes = minutes_acc + to_float(row["usable_minutes_estimate"])
            if clips_acc == 0.0 and minutes_acc == 0.0 and (
                next_clips > MAX_CLIPS_PER_SOURCE or next_minutes > MAX_MINUTES_PER_SOURCE
            ):
                row["reasons"] = agregar_reason(row["reasons"], "source_cap_applied_partial")
                row["recommended_use"] = "ingest_partial"
                row["accepted_clips_estimate"] = min(to_float(row["accepted_clips_estimate"]), MAX_CLIPS_PER_SOURCE)
                row["usable_minutes_estimate"] = min(to_float(row["usable_minutes_estimate"]), MAX_MINUTES_PER_SOURCE)
                clips_acc = to_float(row["accepted_clips_estimate"])
                minutes_acc = to_float(row["usable_minutes_estimate"])
            elif next_clips > MAX_CLIPS_PER_SOURCE or next_minutes > MAX_MINUTES_PER_SOURCE:
                row["reasons"] = agregar_reason(row["reasons"], "source_cap_applied_backup")
                row["recommended_use"] = "manual_review"
                row["decision"] = "maybe_review"
                row["accepted_clips_estimate"] = 0
                row["usable_minutes_estimate"] = 0.0
            else:
                clips_acc = next_clips
                minutes_acc = next_minutes
    return rows


def agregar_reason(reasons: str, reason: str) -> str:
    items = [r for r in reasons.split("|") if r]
    if reason not in items:
        items.append(reason)
    return "|".join(items)


def build_rows(candidates: list[dict[str, str]], audits: dict[str, dict[str, Any]], clips_per_minute: float) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        key = candidate.get("video_id") or ""
        audit = audits.get(key)
        if audit is None:
            audit = next((a for a in audits.values() if a.get("url") == candidate.get("url")), None)
        rows.append(row_desde_candidate(candidate, audit, clips_per_minute))
    rows = aplicar_caps(rows)
    rows.sort(key=lambda r: (decision_orden(str(r["decision"])), -to_float(r["total_score"])))
    return rows


def escribir_shortlist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shortlist = [r for r in rows if r["decision"] in {"strong_accept", "accept", "maybe_review"}]
    shortlist.sort(
        key=lambda r: (
            decision_orden(str(r["decision"])),
            -to_float(r["total_score"]),
            -to_float(r["visual_quality_score"]),
            -to_float(r["usable_minutes_estimate"]),
            str(r["channel"]),
        )
    )
    out = []
    for rank, row in enumerate(shortlist, start=1):
        contact = f"data_pipeline/discovery/outputs/contact_sheets/{row['video_id']}.jpg" if row.get("video_id") else ""
        out.append(
            {
                "rank": rank,
                "url": row["url"],
                "title": row["title"],
                "channel": row["channel"],
                "decision": row["decision"],
                "total_score": row["total_score"],
                "visual_quality_score": row["visual_quality_score"],
                "audio_quality_score": row["audio_quality_score"],
                "context_score": row["context_score"],
                "usable_minutes_estimate": row["usable_minutes_estimate"],
                "accepted_clips_estimate": row["accepted_clips_estimate"],
                "recommended_use": row["recommended_use"],
                "reasons": row["reasons"],
                "contact_sheet_path": contact,
                "notes": row["uncertainty"],
            }
        )
    escribir_csv(SHORTLIST_CSV, out, SHORTLIST_FIELDS)
    return out


def escribir_rechazados(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rejected = [
        {
            "url": r["url"],
            "title": r["title"],
            "channel": r["channel"],
            "decision": r["decision"],
            "total_score": r["total_score"],
            "visual_quality_score": r["visual_quality_score"],
            "audio_quality_score": r["audio_quality_score"],
            "context_score": r["context_score"],
            "reject_reasons": r["reasons"],
        }
        for r in rows
        if r["decision"] == "reject"
    ]
    escribir_csv(REJECTED_CSV, rejected, REJECTED_FIELDS)
    return rejected


def escribir_diversidad(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["decision"] in {"strong_accept", "accept", "maybe_review"}:
            grouped[str(row["channel"] or "unknown")].append(row)
    out = []
    for channel, channel_rows in sorted(grouped.items()):
        accepted_rows = [r for r in channel_rows if r["decision"] in {"strong_accept", "accept"}]
        clips = sum(to_float(r["accepted_clips_estimate"]) for r in accepted_rows)
        minutes = sum(to_float(r["usable_minutes_estimate"]) for r in accepted_rows)
        notes = []
        decision = "ok"
        if clips >= MAX_CLIPS_PER_SOURCE or minutes >= MAX_MINUTES_PER_SOURCE:
            decision = "cap_applied_or_near_cap"
            notes.append("limitar_a_mejores_videos_de_la_fuente")
        if not accepted_rows and channel_rows:
            decision = "backup_review"
        out.append(
            {
                "channel": channel,
                "source_type": moda([r["source_type"] for r in channel_rows]),
                "accepted_videos": len(accepted_rows),
                "usable_minutes_estimate": round(minutes, 2),
                "accepted_clips_estimate": round(clips, 0),
                "avg_total_score": round(mean([to_float(r["total_score"]) for r in channel_rows]), 2),
                "avg_visual_score": round(mean([to_float(r["visual_quality_score"]) for r in channel_rows]), 2),
                "avg_audio_score": round(mean([to_float(r["audio_quality_score"]) for r in channel_rows]), 2),
                "decision": decision,
                "notes": "|".join(notes),
            }
        )
    escribir_csv(DIVERSITY_CSV, out, DIVERSITY_FIELDS)
    return out


def moda(values: list[Any]) -> Any:
    counts: dict[Any, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0] if counts else ""


def escribir_target_progress(rows: list[dict[str, Any]], diversity: list[dict[str, Any]], clips_per_minute: float) -> None:
    accepted = [r for r in rows if r["decision"] in {"strong_accept", "accept"}]
    maybe = [r for r in rows if r["decision"] == "maybe_review"]
    rejected = [r for r in rows if r["decision"] == "reject"]
    clips = sum(to_float(r["accepted_clips_estimate"]) for r in accepted)
    minutes = sum(to_float(r["usable_minutes_estimate"]) for r in accepted)
    distinct_sources = len({r["channel"] for r in accepted if r.get("channel")})
    remaining_clips = max(0.0, TARGET_NEW_ACCEPTED_CLIPS - clips)
    remaining_minutes = max(0.0, TARGET_USABLE_MINUTES_LOW - minutes)
    target_clips_reached = clips >= TARGET_NEW_ACCEPTED_CLIPS
    target_minutes_reached = TARGET_USABLE_MINUTES_LOW <= minutes <= TARGET_USABLE_MINUTES_HIGH
    if target_clips_reached:
        target_decision = "TARGET_REACHED"
    elif minutes >= TARGET_USABLE_MINUTES_LOW:
        target_decision = "TARGET_CLIPS_NOT_REACHED_USABLE_MINUTES_REACHED"
    elif not accepted and any("missing_audit" in r["reasons"] for r in rows):
        target_decision = "TARGET_NOT_REACHED_NEED_AUDIT_SAMPLES"
    elif distinct_sources < 10:
        target_decision = "TARGET_NOT_REACHED_NEED_MORE_SOURCES"
    elif any("audio_proxy_debil" in str(r.get("uncertainty", "")) for r in rows):
        target_decision = "TARGET_NOT_REACHED_ASR_BLOCKED"
    else:
        target_decision = "TARGET_NOT_REACHED_QUALITY_TOO_STRICT"

    top_sources = sorted(diversity, key=lambda r: -to_float(r["accepted_clips_estimate"]))[:10]
    lines = [
        "# Target progress v1",
        "",
        f"target_new_accepted_clips: {TARGET_NEW_ACCEPTED_CLIPS}",
        f"stretch_new_accepted_clips: {STRETCH_NEW_ACCEPTED_CLIPS}",
        f"accepted_new_clips_estimate: {clips:.0f}",
        f"remaining_clips_to_target: {remaining_clips:.0f}",
        f"usable_minutes_estimate: {minutes:.1f}",
        f"remaining_usable_minutes: {remaining_minutes:.1f}",
        f"target_clips_reached: {str(target_clips_reached).lower()}",
        f"target_usable_minutes_reached: {str(target_minutes_reached).lower()}",
        f"accepted_videos_count: {len(accepted)}",
        f"maybe_videos_count: {len(maybe)}",
        f"rejected_videos_count: {len(rejected)}",
        f"distinct_sources_count: {distinct_sources}",
        f"clips_per_minute_estimate: {clips_per_minute:.2f}",
        f"source_caps_applied: max_clips_per_source={MAX_CLIPS_PER_SOURCE}, max_usable_minutes_per_source={MAX_MINUTES_PER_SOURCE}",
        "quality_thresholds: min_total_score=85, min_visual_quality_score=80, min_audio_quality_score=75, accent=argentino/rioplatense_probable",
        f"target_decision: {target_decision}",
        "",
        "## Top sources",
    ]
    for source in top_sources:
        lines.append(
            f"- {source['channel']}: {source['accepted_clips_estimate']} clips, {source['usable_minutes_estimate']} min, decision={source['decision']}"
        )
    lines.extend(
        [
            "",
            "## Recomendacion concreta",
            recomendacion(target_decision, accepted, maybe),
            "",
        ]
    )
    TARGET_PROGRESS_MD.write_text("\n".join(lines), encoding="utf-8")


def recomendacion(target_decision: str, accepted: list[dict[str, Any]], maybe: list[dict[str, Any]]) -> str:
    if target_decision == "TARGET_REACHED":
        return "Ingestar primero los strong_accept/accept respetando caps por fuente; no hace falta bajar thresholds."
    if accepted:
        return "Ingestar primero la shortlist accepted y seguir buscando fuentes nuevas antes de usar backups maybe_review."
    if maybe:
        return "Revisar manualmente los maybe_review con mejor score y ejecutar una segunda ronda de busqueda/auditoria."
    return "Ejecutar mas busquedas y auditorias de samples; con la evidencia actual no hay fuentes aceptadas."


def escribir_review_index(shortlist: list[dict[str, Any]]) -> None:
    lines = ["# Review index", ""]
    for row in shortlist:
        lines.extend(
            [
                f"## {row['rank']}. {row['title']}",
                f"- link: {row['url']}",
                f"- channel: {row['channel']}",
                f"- score: total={row['total_score']}, visual={row['visual_quality_score']}, audio={row['audio_quality_score']}, context={row['context_score']}",
                f"- decision: {row['decision']} / {row['recommended_use']}",
                f"- reasons: {row['reasons']}",
                f"- contact_sheet_path: {row['contact_sheet_path']}",
                "",
            ]
        )
    REVIEW_INDEX_MD.write_text("\n".join(lines), encoding="utf-8")


def escribir_ingest_plan(rows: list[dict[str, Any]]) -> None:
    accepted = [r for r in rows if r["decision"] in {"strong_accept", "accept"}]
    accepted.sort(key=lambda r: (decision_orden(str(r["decision"])), -to_float(r["total_score"])))
    lines = [
        "# Ingest plan v1",
        "",
        "No ejecutar ingest full sin aprobacion explicita. Este plan ordena candidatos para una primera ingesta controlada.",
        "",
        "## Orden recomendado",
    ]
    for idx, row in enumerate(accepted[:30], start=1):
        lines.extend(
            [
                f"### {idx}. {row['title']}",
                f"- url: {row['url']}",
                f"- channel: {row['channel']}",
                f"- source_type: {row['source_type']}",
                f"- score: total={row['total_score']}, visual={row['visual_quality_score']}, audio={row['audio_quality_score']}, context={row['context_score']}",
                f"- estimated accepted clips: {row['accepted_clips_estimate']}",
                f"- estimated usable minutes: {formatear_minutos(to_float(row['usable_minutes_estimate']))}",
                f"- recommended_use: {row['recommended_use']}",
                f"- riesgos: {row['uncertainty']}; {row['reasons']}",
                f"- comando tentativo: python data_pipeline/descargar_procesar.py \"{row['url']}\"",
                "",
            ]
        )
    lines.extend(
        [
            "## Datos a guardar en bucket al ingestar",
            "- metadata de fuente y video.",
            "- video/clips/audio/transcripts si se aprueba la ingesta.",
            "- ROIs derivados despues del preprocesamiento visual.",
            "",
            "ROIs solos no alcanzan para discovery o limpieza de transcripts: faltan audio, contexto y metadata.",
            "",
            "## Comandos para escalar discovery",
            "No buscar llegar a 12K en un loop local largo. Repetir en lotes chicos, guardando outputs livianos y frenando cualquier auditoria sin progreso/logs nuevos por mas de 30 minutos.",
            "",
            "```bash",
            'python -m data_pipeline.discovery.src.search_youtube_candidates --append --max-results 8 --query "podcast argentino entrevista camara fija" --query "entrevista argentina completa"',
            "python -m data_pipeline.discovery.src.audit_youtube_candidate --input data_pipeline/discovery/outputs/candidate_videos_round3_for_audit.csv --limit 15 --run-asr --asr-model small",
            "python -m data_pipeline.discovery.src.score_candidates --clips-per-minute 12",
            "python -m data_pipeline.discovery.src.make_contact_sheets --accepted-only --limit 80",
            "```",
        ]
    )
    INGEST_PLAN_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    configurar_salida_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=CANDIDATE_VIDEOS_CSV)
    parser.add_argument("--sample-metadata", type=Path, default=SAMPLE_METADATA)
    parser.add_argument("--clips-per-minute", type=float, default=DEFAULT_CLIPS_PER_MINUTE)
    args = parser.parse_args()

    asegurar_directorios()
    candidates = [r for r in leer_csv(args.candidates) if r.get("url")]
    audits = cargar_auditorias(args.sample_metadata)
    rows = build_rows(candidates, audits, args.clips_per_minute)
    escribir_csv(CANDIDATE_SCORES_CSV, rows, SCORE_FIELDS)
    shortlist = escribir_shortlist(rows)
    rejected = escribir_rechazados(rows)
    diversity = escribir_diversidad(rows)
    escribir_target_progress(rows, diversity, args.clips_per_minute)
    escribir_review_index(shortlist)
    escribir_ingest_plan(rows)
    print(f"scores={len(rows)} -> {CANDIDATE_SCORES_CSV}")
    print(f"shortlist={len(shortlist)} -> {SHORTLIST_CSV}")
    print(f"rejected={len(rejected)} -> {REJECTED_CSV}")
    print(f"diversity_sources={len(diversity)} -> {DIVERSITY_CSV}")
    print(f"target_progress -> {TARGET_PROGRESS_MD}")
    print(f"ingest_plan -> {INGEST_PLAN_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
