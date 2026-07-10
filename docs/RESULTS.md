# RESULTS — ledger de resultados VSR rioplatense

Documento **vivo**: acá se registran todos los números del proyecto (WER/CER, datos de train,
repos/bases, latencias, crudo vs corregido, etc.). Se va actualizando con cada experimento.
Última actualización: **2026-07-05**.

## Cómo leer esta tabla

- **Test set:** salvo que se diga lo contrario, todos los números propios son sobre **`test-658`**
  = 658 clips de **2 hablantes held-out** (f22/f15) nunca vistos en train → **speaker-independent**.
  El split val/test está **congelado** y es el mismo desde ft03, así que ft03→ft07 son head-to-head.
- **Métrica:** %WER y %CER con **IC 95%** por bootstrap (2000 iters). IC que no se solapan =
  diferencia estadísticamente significativa.
- **Normalización:** minúsculas, sin acentos (la ñ se preserva), sin puntuación. Misma `norm()`
  en todas las evaluaciones (`vsr/mpc001/scripts/zeroshot.py`).
- **Familia de arquitectura (todos los modelos propios):** Conv3D + ResNet18 → Conformer (12 capas)
  → decoder híbrido CTC/Attention. **Offline / bidireccional.** ~102M params (incl. LM).

---

## 1. Modelos propios (fine-tunes sobre rioplatense)

| Modelo | Base pre-entrenada | Repo base | Train (clips) | Dataset aprox. | Config | %WER | %CER |
|---|---|---|---|---|---|---|---|
| **ft03** | LIP-RTVE (español TV) | Gimeno | 4818 | ronda-1 (~14h, 9212 npz) | v1 full-FT | 68.93 ± 1.27 | 41.01 ± 0.96 |
| **ft04** | LIP-RTVE | Gimeno | ~4800 | ronda-1 | v2 freeze+aug | 69.73 ± 1.30 | 42.29 ± 0.95 |
| **ft05** ⭐ | LIP-RTVE | Gimeno | 10934 | ronda-2 (~19h, 12112 npz) | v1 full-FT | **65.05 ± 1.29** | **38.24 ± 0.97** |
| **ft06** | LIP-RTVE | Gimeno | ~10900 | ronda-2 | v2 freeze+aug | 66.37 ± 1.31 | 39.34 ± 0.90 |
| **zero-shot ES** | CMU-MOSEAS ES (multiling.) | mpc001 | 0 (zero-shot) | — | — | 71.50 ± 1.70 | 46.88 ± 1.24 |
| **ViSpeR zero-shot** ⭐⭐ | multilingüe **794h** (SPM 21k, `<es>`) | ViSpeR/TII | 0 (zero-shot) | — | — | **45.22 ± 1.90** | **26.98 ± 1.27** |
| **ViSpeR ft-argentino** | ViSpeR 794h → fine-tune AR | ViSpeR/TII | 8067 | argentino | full-FT | 61.51 ± 1.93 ❌ | 38.59 ± 1.37 |
| **ViSpeR LoRA+aug AR** | ViSpeR 794h → freeze+LoRA+aug | ViSpeR/TII | 8067 | argentino | LoRA r16 | 45.97 ± 2.02 ≈ | 26.00 ± 1.25 |
| **ft05b** | LIP-RTVE | Gimeno | 8067 | same-data | v1 full-FT | 70.30 ± 1.30 | 42.08 ± 0.95 |
| **ft07** | CMU-MOSEAS ES remapeado | mpc001→Gimeno | 8067 | same-data | v1 full-FT | 69.15 ± 1.28 | 41.66 ± 0.91 |

⭐ **ft05 = mejor modelo propio a la fecha** (modelo de producción de referencia).

**Notas sobre los grupos de experimentos:**
- **ft03/ft04 (ronda-1)** y **ft05/ft06 (ronda-2)**: misma receta, distinta cantidad de datos.
  Los train counts reflejan los clips CON transcripción disponibles en cada ronda.
- **ft05b/ft07 (same-data)**: experimento A/B **base-vs-base** con train IDÉNTICO (8067) y mismo
  seed. Compara la base LIP-RTVE (ft05b) contra la multilingüe remapeada (ft07). Los números
  absolutos NO igualan a ft05 (65.05) porque el train es menor (8067 vs 10934).
- **v1** = full fine-tuning; **v2** = frontend congelado + RandomCrop(88) + HorizontalFlip(0.5).

**Conclusiones (de nuestras corridas):**
1. **Más datos = mejora significativa.** ft05 vs ft03: −3.9 WER (IC no solapan). ft06 vs ft04: −3.4 WER.
2. **v1 (full-FT) ≥ v2 (freeze+aug)**, pero la diferencia NO es significativa (IC solapan).
3. **Base multilingüe ≈ base LIP-RTVE** con datos idénticos: ft07 apenas mejor que ft05b
   (−1.16 WER, −0.42 CER) pero IC solapan → empate técnico. **Los datos importan mucho más que la base.**
4. Fine-tunear (ft05 65.05) supera claramente al multilingüe zero-shot (71.50) → adaptar al acento paga.

---

## 2. Benchmarks externos (referencia — OTROS test sets, NO comparables directo)

Contexto del estado del arte. **Ojo: cada número es sobre su propio test set**, no sobre `test-658`.
Sirven para ubicar nuestros números, no para comparar 1:1. Todos offline/bidireccionales salvo aclaración.

| Sistema | Repo | Dataset (naturaleza) | Métrica | Datos train | Params |
|---|---|---|---|---|---|
| Auto-AVSR (VSR) | mpc001/auto_avsr | LRS2 (TV BBC, EN) | 14.6% WER | 3448 h | 250M |
| Auto-AVSR (VSR) | mpc001/auto_avsr | **LRS3 (charlas TED, EN)** | **19.1% WER** | 3448 h | 250M |
| Auto-AVSR (VSR) | mpc001/auto_avsr | LRS3, solo 818h | 36.3% WER | 818 h | 250M |
| mpc001 multilingüe | mpc001/VSR-multi | LRS3 (TED, EN) | 19.1% WER | — | ~50M |
| mpc001 multilingüe | mpc001/VSR-multi | CMLR (noticiero TV, ZH) | 8.0% CER | — | ~50M |
| mpc001 multilingüe | mpc001/VSR-multi | **CMU-MOSEAS ES (YouTube)** | **43.9–44.5% WER** | 16 h | ~50M |
| mpc001 multilingüe | mpc001/VSR-multi | CMU-MOSEAS PT / FR | 51.4% / 58.6% WER | — | ~50M |
| Gimeno | evaluating-end2end-spanish | VLRF (laboratorio, ES, 24 hab) | 24.8% WER (spk-dep) | 3 h | ~102M |
| Gimeno | evaluating-end2end-spanish | **LIP-RTVE (noticiero RTVE, ES, 323 hab)** | **34.5% dep / 59.5% indep** | 13 h | ~102M |
| Gimeno | evaluating-end2end-spanish | CMU-MOSEAS ES / MuAViC ES | 44.6% / 56.3% WER | — | ~102M |
| PyTorch AV-ASR (**streaming**) | pytorch blog | LRS3 (TED) — **audio+video** | 1.6–2.6% WER | — | 35–383M |

**Referencia de ingeniería (no benchmark):** **Chaplin** (`amanvirparhar/chaplin`) usa el checkpoint
Auto-AVSR LRS3 (19.1%) + MediaPipe + corrección **Qwen3-4B (Ollama)**. NO es streaming: es
push-to-talk (grabar-soltar, inferencia offline sobre el clip entero). Es el blueprint de nuestro demo.

**Lecturas clave:**
- **Nadie tiene VSR visual-only en streaming** (el único streaming real es audio-visual).
- **Datos y acento dominan:** LRS3 19.1% (EN) → CMU-MOSEAS 44.5% (ES) → rioplatense 71.5% zero-shot.
- LRS3 sube de 19.1% a 36.3% al bajar de 3448h a 818h. **Nosotros tenemos ~14–19h.**
- Nuestro análogo real es **LIP-RTVE speaker-independent (59.5% WER)**: ft05 (65) está a ~5 pts,
  con menos datos y un acento más lejano → resultado defendible.

Fuentes: Auto-AVSR arXiv:2303.14307 · mpc001 arXiv:2202.13084 · Gimeno arXiv:2502.00464 ·
Chaplin github.com/amanvirparhar/chaplin · PyTorch real-time AV-ASR blog.

---

## 3. Fase 0 — corrección con LLM (crudo vs corregido)

**Setup:** corrector = `qwen3:4b-instruct-2507-q4_K_M` (Ollama local, no-razonador, Q4_K_M).
System-prompt = corrector de español rioplatense (voseo), conservador (no agrega info).
Se corre sobre las hipótesis ya generadas (`test.inf`, ref#hyp) y se re-mide con la misma `norm()`.
Script: `vsr/mpc001/scripts/fase0_llm_correct.py`. Se estratifica por CER-por-clip del baseline
para testear la hipótesis: *CER bajo → el LLM mejora; CER alto → el LLM alucina y empeora*.

**Resultado (2026-07-05, corrector `qwen3:4b-instruct-2507-q4_K_M`, 658 clips, ~1.27 s/clip CPU/Mac):**

| Modelo | %WER crudo | %WER corregido | Δ WER | %CER crudo | %CER corregido | Δ CER |
|---|---|---|---|---|---|---|
| ft05b | 70.30 | 71.04 | **+0.74** ❌ | 42.07 | 43.77 | **+1.70** ❌ |
| ft07 | 69.15 | 70.11 | **+0.96** ❌ | 41.65 | 43.26 | **+1.61** ❌ |

**Estratificación por CER-por-clip del baseline** (hipótesis: CER bajo → mejora; CER alto → empeora):

| Rango CER | ft05b n / Δ WER | ft07 n / Δ WER |
|---|---|---|
| 0–20 | 49 / +0.99 | 49 / +0.00 |
| 20–40 | 239 / +1.11 | 241 / +1.46 |
| 40–60 | 297 / +0.58 | 301 / +0.78 |
| 60–200 | 73 / −0.10 | 67 / +0.54 |

**Conclusión (resultado NEGATIVO, robusto en ambos modelos):** la corrección LLM post-hoc **naive
NO ayuda — empeora levemente el WER y más el CER, en TODOS los estratos.** Ni siquiera el bucket más
limpio (CER 0–20) mejora (a lo sumo empata). Razón: incluso el bucket "CER bajo" tiene WER ~34% → no
son frases casi correctas que el LLM pula, sino texto ya roto; el modelo sobre-corrige (cambia palabras
correctas por incorrectas) más de lo que arregla. **No tenemos clips con CER genuinamente bajo (~5%)**
donde la hipótesis se podría confirmar. Mensaje central: **con CER ~42 no hay señal suficiente para que
la corrección gane → bajar el CER es prioritario sobre agregar el LLM.**

**Limitaciones de este experimento (qué podría rescatar la idea):** (1) input ya normalizado — se le
quitan acentos/puntuación/mayúsculas, las pistas que un corrector usa, y la métrica es ciega a los
acentos que sí arregla; (2) solo 1-best, sin n-best ni contexto de caption previo; (3) modelo chico
(4B) y zero-shot, sin few-shot rioplatense; (4) qwen3 base "thinking" NO sirve (razona en el output y
rompe todo → usar variante `instruct`). Un rediseño (n-best + contexto + few-shot + modelo mayor +
conservar ortografía) podría cambiar el signo, pero el techo lo pone el CER.

**RE-TEST con CER más bajo — ViSpeR zero-shot (CER 27, vs 42 de ft05b/ft07) — 2026-07-06:**

| Modelo | %WER crudo | %WER corr | Δ WER | %CER crudo | %CER corr | Δ CER |
|---|---|---|---|---|---|---|
| **ViSpeR zero-shot** (CER 27) | 45.22 | 46.64 | **+1.42** ❌ | 26.98 | 28.32 | **+1.35** ❌ |

Estratos por CER-por-clip: **0–20: +2.26** (¡el que MÁS empeora!) · 20–40: +1.11 · 40–60: +0.64 · 60+: 0.00.

**CONCLUSIÓN DEFINITIVA (hipótesis del CER REFUTADA):** aun bajando el CER de 42 a 27, y aun en los clips
MÁS limpios (CER 0–20, WER ~22), el corrector qwen naive **empeora** — y empeora MÁS donde el CER es bajo.
El modelo sobre-corrige: cambia palabras ya correctas por formas genéricas. Con dos regímenes de CER el
resultado es consistente → **la corrección LLM post-hoc naive NO ayuda al VSR**, independientemente del CER.
Para que un LLM sirviera haría falta otro diseño (corrector fine-tuneado a la tarea, n-best rescoring) o un
CER mucho menor (<10). Cierra el experimento del componente agéntico: el valor está en el modelo visual, no
en un post-corrector genérico.

---

## 4. Latencia / eficiencia (Fase 1)

RTF (real-time factor) y latencia por chunk, desglosado por etapa (frontend / encoder / decoder),
barriendo decoding {beam+LM, CTC-greedy} × quantización {fp32, int8/fp16} × {CPU, GPU}.

> ⏳ **PENDIENTE** (Fase 1). Se completa cuando se corra el harness de benchmark de inferencia.

| Config | Etapa | RTF | Latencia/chunk | %WER | Notas |
|---|---|---|---|---|---|
| _pend._ | | | | | |

---

## 5. Data AV española disponible (para escalar — currículum de pre-entrenamiento)

Para VSR los visemas son casi dialecto-agnósticos → español general sirve para pre-entrenar; lo argentino
rinde en el fine-tune final + test. Data AV española **con video usable (píxeles)** disponible hoy:

| Corpus | Horas ES | Video? | Transcripción | Acceso | Nota |
|---|---|---|---|---|---|
| **ViSpeR** | **794** (207 TEDx + 587 wild) | ✅ | ✅ | HF `tiiuae/visper`, CC-BY-NC | el más grande; link-rot en las wild |
| **MuAViC** | **178** (~100-140 real) | ✅ | ✅ | github facebookresearch/muavic | mismo preproc que nosotros; repo archivado |
| **LIP-RTVE** | ~13 (→24) | ✅ | ✅ | github + NDA RTVE | ya lo tenemos (base de ft05) |
| CMU-MOSEAS ES | <20 | ❌ solo features | ✅ | CMU SDK | **inútil** (no da píxeles) |
| VLRF | ~1-3 | ✅ | ✅ | página autor | lab peninsular; ya lo tenemos |
| AVSpeech ES / VoxCeleb2 ES | ~204 / ~42 | ✅ | ❌ | links YT | requieren pseudo-label (Whisper) |

**Disponibilidad medida (2026-07-05, `yt-dlp --simulate` sobre muestra aleatoria de IDs):**

| Corpus | IDs totales | Muestra | Vivos | Tasa viva | Nominal | Recuperable est. |
|---|---|---|---|---|---|---|
| **ViSpeR-es** (train) | 14.882 videos | 250 | 216 OK / 4 muertos / 30 rate-limit | **98.2%** | 794 h | **~700–780 h** |
| **MuAViC-es** | ~190 (muestra sesgada) | 150 | 150 OK / 0 muertos | **100%** | 178 h | **~150–178 h** |

Caveat: `--simulate` verifica existencia+metadata, es **techo** (la descarga real puede fallar por
age-restrict/geo/formato). Muestra MuAViC sesgada al inicio del archivo. Aun así: **la data está viva.**
**Techo con texto listo ≈ 985h nominal → realista ~700–900h descargables hoy** (vs ~19h rioplatenses).
**No existe corpus AV rioplatense/LatAm dedicado** → dialecto argentino nativo solo vía pipeline propio.
Todo requiere bajar de YouTube + croppear + GPU. Detalle en memoria `datasets-spanish-av-video`.

## 6. ViSpeR — hallazgo clave (2026-07-05 noche)

**ViSpeR zero-shot (modelo multilingüe 794h, sin fine-tune argentino) = WER 45.22 / CER 26.98 sobre
test-658** → le gana a NUESTRO MEJOR modelo (ft05 = 65.05, fine-tuneado) por ~20 pts de WER, y a mpc001
zero-shot (71.5) por ~26 pts. El encoder de 794h de español transfiere muchísimo mejor al rioplatense que
partir de LIP-RTVE (13h) o CMU-MOSEAS (16h). Mantiene su tokenizer SPM 21k (language token `<es>`).
Esto reorientó la estrategia: en vez del currículum-desde-cero (ft08, train stage1 sobre 50h), **partimos
de ViSpeR y lo fine-tuneamos sobre argentino**. Harness: `~/Desktop/visper/visper_zeroshot.py`
(env conda `visper`, CPU ~5.9s/clip). Modelo `visper_vsr_base.pth`. Licencia CC BY-NC (uso research OK).

### Fine-tune de ViSpeR sobre argentino (2026-07-06) — EMPEORA (overfit)

Fine-tune full-FT del modelo ViSpeR sobre nuestras ~19h argentinas (8067 clips, batch1+accum8, su SPM),
en L4. **Resultado: WER 61.51 / CER 38.59 sobre test-658 — PEOR que el zero-shot (45.22).** El val_loss
subió monótono desde época 1 (39.05 → 39.71 → 40.06 → 41.16), overfit clásico; el early-stopping guardó
época 1 (el menos sobreajustado) e igual quedó 16 pts de WER por encima del zero-shot. **Con ~19h no
alcanza para adaptar un modelo de 794h sin degradarlo** (catastrophic forgetting / sobreajuste al set chico).
Modelo en `modelos/ft_visper_ar_best.pth`. Scripts: `~/Desktop/visper/fine_tune_visper.py` + `dws_startup_visper.sh`.

---

## 7. RESUMEN FINAL — comparación de todos los enfoques (test-658, rioplatense)

| # | Enfoque | %WER | %CER | Nota |
|---|---|---|---|---|
| 1 | mpc001 zero-shot (CMU-MOSEAS ES 16h) | 71.50 | 46.88 | base multilingüe chica |
| 2 | ft03/ft04 (LIP-RTVE + ~9h AR) | 68.9 / 69.7 | ~41/42 | ronda-1 |
| 3 | ft05b / ft07 (same-data 8067) | 70.3 / 69.2 | ~42 | A/B de bases |
| 4 | **ft05** (nuestro mejor, LIP-RTVE + ~19h AR) | **65.05** | 38.24 | mejor propio |
| 5 | ViSpeR ft-argentino (794h → +19h AR, **full-FT**) | 61.51 | 38.59 | overfit, peor que zs |
| 6 | **ViSpeR zero-shot (794h)** ⭐ | **45.22** | 26.98 | **el mejor / más simple** |
| 7 | ViSpeR **LoRA+aug** AR (freeze+LoRA r16+augment) | 45.97 ≈ | **26.00** | empata zs (IC solapan) |
| 8 | ViSpeR zero-shot + corrector qwen | 46.64 | 28.32 | el LLM empeora |

**Conclusiones del proyecto:**
1. **El zero-shot de ViSpeR es el mejor punto de partida** (45.22 WER): −20 vs nuestro mejor fine-tune propio
   (ft05 65.05), −26 vs mpc001. Domina la **escala del pre-entrenamiento visual español** (794h), no la
   arquitectura (todas comparten Conv3D+ResNet18→Conformer→CTC/Att).
2. **La forma de fine-tunear importa muchísimo, pero 19h no alcanzan para mejorar el zero-shot:**
   - *full-FT* con ~19h → **61.51 (lo destroza, overfit).**
   - *freeze+LoRA (0.8% params)+augment+early-stop* → **45.97 / CER 26.00: EMPATA el zero-shot** (IC solapan;
     WER +0.75, CER −0.98, dentro del ruido). O sea: la regularización correcta **evita la degradación** del
     full-FT, pero el dataset argentino chico **no agrega señal** por encima de lo que ya capturó el pre-entreno
     de 794h. El CER baja un pelín (26.0 vs 27.0) — sugestivo pero no significativo.
   - **Implicancia:** el dataset NO fue en vano — es el benchmark (test-658) que reveló todo, y mostró que
     harían falta **muchas más horas argentinas** (o el mismo LoRA con +datos) para superar el zero-shot.
3. **El corrector LLM post-hoc naive (qwen3:4b) NO ayuda** a ningún nivel de CER (42 ni 27): sobre-corrige y
   empeora el WER, peor cuanto más limpio el clip. El valor está en el modelo visual.
4. **Modelo de producción recomendado: ViSpeR zero-shot** (`visper_vsr_base.pth`, token `<es>`) — el LoRA+aug
   empata y es más complejo, así que para producción conviene el zero-shot directo. Próximo paso para tiempo
   real: **ingeniería de streaming** (sliding-window tipo Chaplin), no más entrenamiento. Licencia CC BY-NC (research).

## Changelog

- **2026-07-05** — Doc creado. Cargados: modelos propios ft03–ft07 + zero-shot + A/B ft05b/ft07;
  benchmarks externos; **Fase 0 completa** (corrección LLM = resultado negativo en ambos modelos);
  **data AV española disponible** (ViSpeR 794h / MuAViC 178h el hallazgo grande). Fase 1 (latencia) pendiente.
- **2026-07-05 (noche)** — **Currículum Fase 1a: 50.03h de ViSpeR-es procesadas** (936 videos, 36.631 clips,
  28GB) vía landmarks ViSpeR → npz 96×96 compatibles con ft05. En bucket `curriculum_visper/lip_rois/`.
  Próximo: run DWS `ft08` (LIP-RTVE → stage español ViSpeR → fine-tune argentino → eval test-658) vs ft05=65.05.
- **2026-07-06 (madrugada)** — **ViSpeR zero-shot = 45.22 WER (mejor de todo el proyecto, −20 vs ft05).**
  Fine-tune de ViSpeR sobre 19h argentino EMPEORÓ (61.51, overfit). Corrector qwen sobre ViSpeR zs EMPEORÓ
  (+1.42, hipótesis CER refutada def.). Resumen final en §7. Teardown completo (0 VMs/discos). Imagen
  `labios-img-visper` preservada. Modelos en `modelos/` (ft_visper_ar_best.pth) + `~/Desktop/visper/visper_vsr_base.pth`.
