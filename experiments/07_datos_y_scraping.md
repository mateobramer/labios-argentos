# 07 — Datos: corpus, dataset argentino y scraping

## A. Dataset argentino propio (el que tenemos)

- Recolectado de YouTube (fuentes rioplatenses talking-head) + Whisper (word-timestamps) + crop MediaPipe.
- Pipeline: `descargar_procesar.py` (descarga+Whisper+segmentación) → `preprocessing/src/preprocesar.py`
  (crop mean-face 96×96 25fps) → npz. QC en `data_cleaning/`.
- **~12.112 npz (~19h)**; splits congelados: **train 8067 / val 466 / test 658** (test = 2 hablantes f15/f22).
  En `gs://labios-argentos-vsr-dataset/{lip_rois,splits}`.
- **candidatos_v2_FINAL.csv** = 32 fuentes de ronda-2 (base de ft05). ~85-971 carpetas ya procesadas.

## B. Dataset "clean-v1" de Martín (compañero)

Bucket aparte `gs://labios-argentos-vsr-clean-v1` (proyecto `labios-argentos-499900`, lectura para el
usuario). Contiene:
- **argentina/existing**: nuestros 12.112 clips re-empaquetados (splits ya asignados).
- **argentina/new_discovery**: **13.193 clips nuevos** (mp4 con audio) de 20 videos; solo **2.248 con ROI
  hecho** (10.945 `blocked_roi_no_face`); transcripciones ASR crudo (el "clean_gpt" NO se aplicó nunca).
- **spanish_general**: ~42.599 npz español general (para currículum).
El handoff `docs/HANDOFF_ROIS_FINETUNE.md` (ft09) usa existing 8067 + los 2248 new_discovery listos.

## C. Corpus AV con licencia (para escalar — currículum)

Para VSR los visemas son casi dialecto-agnósticos → español general sirve para pre-entrenar.
Disponibilidad medida (`yt-dlp --simulate`, 2026-07-05):

| Corpus | Horas ES | Video | Acceso | Recuperable est. |
|---|---|---|---|---|
| **ViSpeR** | 794 (207 TEDx + 587 wild) | ✅ | HF, CC-BY-NC | ~700-780h |
| **MuAViC** | 178 | ✅ | github facebookresearch/muavic | ~150-178h |
| LIP-RTVE | ~13 | ✅ | github + NDA | base de ft05 |
| CMU-MOSEAS ES | <20 | ❌ solo features | — | inútil |

**No existe corpus AV rioplatense/LatAm público dedicado** → el acento nativo solo vía pipeline propio.

## D. ⚠️ Scraping masivo de YouTube — PARED (2026-07-09)

Intento de ampliar el dataset: enumerados **303 videos / 72.5h** de 18 canales talking-head. El pipeline
local **funciona** (1 video Paulina = 86 clips). PERO:
- Con las cookies del usuario (`~/yt_cookies.txt`) + throttle respetuoso, **YouTube invalidó la sesión a
  los ~30 min** → todo pasó a "Sign in to confirm you're not a bot". Re-exportar cookies moriría igual.
  Seguir insistiendo **arriesga la cuenta de Google**.
- **Rotación de IPs / proxies = evasión de detección → NO se hace.** La barrera de seguridad de Claude
  Code además bloquea `--cookies-from-browser` (usar siempre `--cookies <archivo>` exportado por el usuario).
- **Yield por fuente varía mucho:** cocina/lifestyle ~5% (planos de comida), comediantes/standup/streamers
  de charla mucho mayor → priorizar talking-head.
- Costo del intento: **US$0** (nada llegó a GCP; scrape local muerto sin subir nada).

**Conclusión:** el scraping masivo de YouTube NO es una vía viable. Caminos legítimos a "más datos":
(1) corpus con licencia (ViSpeR/MuAViC) para currículum; (2) los 13k de Martín (solo faltan ROIs);
(3) grabaciones con consentimiento (el self-test funcionó, ver [05](05_selftest_limpio.md)).

Scripts del intento: scratchpad `scrape/` (`build_targets.py`, `run_scrape.sh`). Ver memoria `youtube-scraping-wall`.
