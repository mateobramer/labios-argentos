# PLAN — currículum de datos (español general → argentino)

Objetivo: usar la data AV española disponible (~700–900h: ViSpeR + MuAViC, ver `RESULTS.md` §5) como
etapa de pre-entrenamiento antes del fine-tune argentino, para mejorar el **teacher** (ft05, WER 65.05).
Filosofía: **validar barato antes de comprometer GPU/disco**. Cada fase tiene un gate go/kill explícito.

## Hipótesis y criterio de éxito

- **Hipótesis:** los visemas son casi dialecto-agnósticos → sumar cientos de horas de español general
  antes del fine-tune argentino baja el WER en test-658 (rioplatense). Evidencia previa: LRS3 cae de
  19% a 36% WER solo por bajar de 3448h a 818h → las horas mandan.
- **Baseline a batir:** ft05 = **65.05 WER / 38.24 CER** (test-658, mismo test congelado).
- **Éxito (gate Fase 1):** el currículum baja el WER de forma **significativa** (IC 95% no solapa con ft05).
  Si no mueve o empeora → kill, no escalar a la Fase 2.

## Setup de datos (lo que ya sabemos)

- **Preproc:** nuestro `video_process.py` = MediaPipe → alineación mean-face (`20words_mean_face.npy`) →
  crop boca **96×96 gris, 25fps**. **MD5 del mean-face IDÉNTICO al de MuAViC** (confirmado en su `utils.py`)
  → MuAViC es drop-in de nuestro preproc.
- **ViSpeR (formato confirmado 2026-07-05):** auto-contenido. `videos_ids_train_es.txt` = 14.882 IDs +
  `spanish_train.tar.gz` (1.48 GB) con JSONs `tedx_chunk_*.json` / `wild_chunk_*.json` **keyed por video ID**;
  cada valor = lista de segmentos con `start`, `end` (seg), `label` (transcripción — ¡sin Whisper!) y
  `landmarks` (5 puntos estables/frame, normalizados). Cableado: **video → cortar [start,end] → nuestro
  `preprocesar.py` (MediaPipe→warp mean-face→96×96 gris) → npz + .txt**. Usamos NUESTRO MediaPipe (no sus
  landmarks) para crops idénticos a nuestro dataset argentino. Descarga real ~53 MB / ~15s por video (360p).
- **MuAViC:** IDs = stems de los `.pkl` en `es_metadata.tgz`; `get_data.py --src-lang es` baja de YouTube
  y croppea. Trae transcripción humana. ~178h.
- **Ambos traen texto** → NO necesitamos Whisper (a diferencia de AVSpeech/VoxCeleb2).
- **Disponibilidad medida (2026-07-05):** ViSpeR-es 98.2% vivo, MuAViC-es ~100% vivo.

## Diseño del entrenamiento (currículum)

Comparación controlada (una sola variable = la etapa española):
- **Baseline:** base LIP-RTVE → fine-tune argentino (= ft05, ya lo tenemos).
- **Currículum:** base LIP-RTVE → **+etapa español (ViSpeR/MuAViC)** → fine-tune argentino → eval test-658.
- Mismo test-658 congelado, mismo seed. `fine_tune.py` entrena desde checkpoint sobre un set de npz, así
  que "etapa española" = entrenar sobre npz español; "fine-tune argentino" = continuar desde ese checkpoint
  sobre nuestros npz argentinos.
- Alternativa (si el currículum secuencial no rinde): **pool mixto** español+argentino con el argentino
  upweighted. Se prueba solo si el secuencial falla.
- Nota endgame: esto produce un **teacher** mejor (bidireccional); el student causal para tiempo real se
  destila DESPUÉS (ver `realtime-vsr-plan`).

---

## FASE 0 — validación barata (local, sin GPU, ~$0)

Objetivo: resolver los unknowns y medir rendimiento ANTES de escalar. **Subset ~30 videos ViSpeR-es**
(MuAViC se pospone a Fase 1: sus segmentos vienen de mTEDx, no auto-contenidos; ViSpeR es el corpus grande,
vivo al 98% y auto-contenido → validamos con él primero).

Progreso probe (2026-07-05): descarga real OK (~53MB/~15s por video); formato ViSpeR entendido
(JSON con start/end/label/landmarks); pipeline `preprocessing/preprocesar.py` ubicado (espera clips
segmentados → produce npz 96×96). Adapter `vsr/curriculum/visper_a_clips.py` escrito y probado end-to-end:
corta segmentos + escribe .txt + genera npz **formato correcto (T,96,96) uint8**.

**HALLAZGO Fase 0 (5 videos, 28 clips):** con nuestro MediaPipe re-detectando sobre el video crudo,
el **yield es solo ~21%** (6/28 ok). Descartes: 13 con ratio=0 (cara nunca detectada — planos sin cara
frontal / cara chica que RetinaFace de ViSpeR sí agarró) + resto con ratio 0.15–0.74 (cae por umbral 80%).
**Solución (Fase 0.b): usar los landmarks que ViSpeR ya provee** (5 puntos/frame → mapeados a los 4 de
`VideoProcess`: ojoD, ojoI, nariz, media(bocaIzq,bocaDer)) para el warp mean-face directo, sin MediaPipe.

**FASE 0.b RESUELTA (2026-07-05) — cropper `vsr/curriculum/visper_crop_landmarks.py`:**
- **Yield 100% (28/28 clips)** sobre los mismos 5 videos (vs 21% con MediaPipe).
- **Crops verificados visualmente:** bocas centradas, gris 96×96, calidad comparable a data VSR estándar
  (formato npz (T,96,96) uint8 idéntico al de ft05).
- Métricas para dimensionar: descarga ~53MB/~15s por video (360p); **disco npz ≈ 480 MB/h**
  (100h≈47GB, 800h≈375GB); crop CPU rápido.
- **GATE FASE 0: VERDE.** Preproc compatible, yield ~100%, rendimiento viable. Listo para Fase 1.

Pasos:
1. **Descargar 30 videos ViSpeR-es + 30 MuAViC-es** (muestra aleatoria de los IDs ya extraídos, en
   `directorio temporal/{visper_ids,muavic_ids}/`) con yt-dlp, local en la Mac (gratis).
2. **Correr `video_process.py`** sobre los crudos → npz 96×96 gris 25fps. Verificar:
   - salen limpios (dimensiones, fps, alineación mean-face correcta — inspección visual de 5-6 crops);
   - los segmentos/transcripciones alinean con el audio;
   - los npz cargan en un **smoke de `fine_tune.py`** (1 batch, forward+backward).
3. **Medir para extrapolar el costo del run completo:**
   - horas usables (clips) por video descargado (raw vs clip);
   - disco por hora de npz; tiempo de descarga+crop por video;
   - % de descargas que fallan de verdad (no --simulate, descarga real → corrige el techo de RESULTS §5).
4. **Resolver:** qué contiene exactamente `spanish_train.tar.gz` de ViSpeR (¿crops/labels/landmarks?),
   y el formato de segmentación → decide si re-procesamos de crudo o reusamos algo.

**Gate Fase 0:** si los npz salen compatibles y el rendimiento (h/video, disco, tiempo) hace viable juntar
≥50h → seguir. Si el preproc no reproduce crops compatibles o el yield real es malísimo → replantear.

**Costo:** ~$0 (local). Wall-time: unas horas.

---

## FASE 1 — prueba de currículum a escala chica (1 GPU run)

Objetivo: ¿un stage español de **~50–100h** ya mueve el WER? Es el go/kill barato del enfoque.

Pasos:
1. Procesar un subset de **~50–100h** español (ViSpeR + MuAViC), respaldado al bucket
   `gs://labios-argentos-vsr-dataset`. Descarga+crop en CPU (local overnight o VM e2 barata).
2. **1 corrida DWS L4** (como ft05/ft07): base LIP-RTVE → stage español (npz) → fine-tune argentino
   (nuestros ~19h) → eval test-658. Marcadores + autoteardown + cron de monitoreo (receta ya probada).
3. Comparar WER/CER vs ft05 (IC 95%). Cargar a `RESULTS.md` como `ft08-curriculum`.

**Gate Fase 1 (go/kill):** WER baja significativo vs 65.05 → ir a Fase 2. Si no → kill (documentar el
negativo: "50-100h de español general no alcanzan"), y redirigir esfuerzo (más datos argentinos / tiempo real).

**Costo estimado:** ~1 día-L4 (DWS). On-demand g2-standard-8 ≈ US$0.7–1/h, Spot ≈ US$0.25/h → **~$6–24**.
Procesamiento CPU: gratis (local) o VM e2 barata. Borrar VM+disco al terminar (regla de siempre).

---

## FASE 2 — run completo (solo si Fase 1 da verde)

Objetivo: escalar al máximo de español disponible (~700–900h) + fine-tune argentino.

Pasos:
1. Descarga+crop masivo de ViSpeR+MuAViC-es. Esto es lo pesado: ~15k videos → necesita **fleet de CPU**
   (varias VMs e2 en paralelo, o tandas locales largas) + **disco ≥500GB–1TB** para los npz. Respaldo incremental al bucket.
2. Entrenamiento con la data completa (varios días-L4 vía DWS; posible multi-etapa).
3. Eval test-658 → `ft09-curriculum-full` en `RESULTS.md`. Snapshot del dataset ampliado.

**Costo estimado (grueso, a refinar con los números de Fase 0):** procesamiento el grueso del costo
(fleet CPU + disco) — del orden de decenas de US$; entrenamiento varios días-L4 (~$20–100 según spot/on-demand).
Se dimensiona con precisión recién con las mediciones de Fase 0.

---

## Riesgos y mitigaciones

- **Descarga real < --simulate** (age/geo/formato): se mide en Fase 0 con descarga real; corrige la estimación.
- **Dilución de dialecto** (mezclar español general): mitigado por el fine-tune argentino final (re-especializa).
- **Costo GPU/stockout L4:** DWS resize-request (ya resuelto antes); teardown + monitoreo por cron.
- **Disco:** Fase 2 necesita ≥500GB; separar preproc (CPU) de train (GPU) en discos distintos (aprendizaje ft05/06).
- **Licencias CC-BY-NC:** uso research/no-comercial OK; no redistribuir crops.

## Estado

- [x] Data disponible medida (RESULTS §5)
- [x] **Fase 0 — validación subset: VERDE** (yield 100% vía landmarks ViSpeR; crops OK; ~480MB/h)
- [x] **Fase 1a — datos procesados: 50.03h ViSpeR-es** (936 videos, 36.631 clips/npz, 28GB, solo 4 fallos
  de descarga; local en `data/processed/lip_rois/visper_*` + bucket `curriculum_visper/lip_rois/`).
  Procesado 2026-07-05 en ~5h con `vsr/curriculum/procesar_visper.py` (throttling de YouTube manejado con
  timeout duro + sleep + backups espaciados; worker nohup-desacoplado para sobrevivir kills del harness).
- [ ] **Fase 1b — run DWS entrenamiento ft08 (próximo paso, gate go/kill vs ft05=65.05)**
- [ ] Fase 2 — run completo
