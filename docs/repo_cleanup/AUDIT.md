# AUDIT — auditoría del repo previa a la limpieza (2026-07-10)

Branch: `chore/repo-cleanup-safe-v2`, creada desde `origin/main` = `a11f0827`
("Merge pull request #27 from mateobramer/chore/reorganizacion").

Metodología: clone parcial (`--filter=blob:none` + sparse-checkout sin `data/` ni
`dataset/`), inspección del árbol completo vía API de GitHub (29.395 archivos, 3.05 GB
en el tree de main), lectura de docs núcleo, greps de paths/buckets/imports,
`python3 -m compileall` (OK, exit 0), y diff contra todas las ramas remotas.

---

## 1. Árbol resumido de main y propósito de cada módulo

| Path | Peso (tree) | Propósito inferido | Estado |
|---|---|---|---|
| `data/` | 2.743 MB (17.182 f) | dataset generado: `clips/` (mp4+txt, ES el dataset), `videos/` (9 crudos), `metadata/` (manifests), `corpus/` | **canónico**, intencional (ver §6) |
| `dataset/` | 248 MB (11.901 f) | clips `keep` post-curación (mp4+txt por video fuente) | canónico |
| `vsr/evaluation/` | 22.6 MB | eval WER/CER contra test-658, parches Gimeno, notebooks 06-07 | canónico |
| `cleaning/visual_quality/` | 21.1 MB | detección de clips malos, notebooks 01-05 y 08 | canónico |
| `vsr/historical/ronda2/` | 7 MB | corrida histórica ronda 2 (ft03–ft07) | **histórico**, sin README |
| `preprocessing/` | 3.7 MB | clips → ROI boca 96×96 (`.npz`), notebook 09 | canónico |
| `vsr/` | 2.9 MB | fine-tuning 50M Gimeno + **splits congelados** | canónico |
| `cleaning/transcript_segmentation/` | 0.4 MB | re-segmentado oracional, notebooks 01-03 propios, tests | canónico |
| `demo/` | 0.1 MB | demo web/ptt/stream + infer_server + calibración | canónico, sin README |
| `docs/` | 0.1 MB | SPEC, ESTRUCTURA, RESULTS (ledger), PLAN_CURRICULUM, archivo/ | canónico |
| `experiments/` | 0.1 MB | índice + 10 docs de experimentos con tabla maestra | **fuente de verdad de resultados** |
| `vsr/mpc001/` | ~0 | notas/scripts base mpc001 (clon externo no versionado) | canónico, sin README |
| `vsr/curriculum/` | ~0 | procesamiento ViSpeR-es para currículum | pausado (ver PLAN_CURRICULUM), sin README |
| `data_pipeline/sources/` | ~0 | CSVs de fuentes curadas (gate 0) | canónico |
| `docs/bibliografia/` | 0.1 MB | 1 PDF de paper de referencia | histórico, sin README |
| raíz | — | README, AGENTS, CLAUDE, TO-DO, data_pipeline/descargar_procesar.py, requirements.txt, .gitignore | canónico |

**Contexto clave**: main fue reorganizado HOY (PR #27). La estructura ya es coherente y
está documentada en `docs/ESTRUCTURA.md`. Esta limpieza NO necesita re-mover módulos:
necesita portar material único de ramas, cerrar huecos de docs y desacoplar paths.

## 2. Duplicaciones detectadas

- En `feature/full-clean-release`: `README_DATASET.md`, `OPEN_ITEMS_DATASET.md` en raíz
  son **byte-idénticos** a sus copias en `data_pipeline/release/reports/`. `HOW_TO_USE_BUCKET.md`
  difiere entre raíz y reports (verificar cuál es más nuevo al portar).
- En main: no se detectaron duplicados relevantes (la reorg de PR #27 ya fusionó
  FLUJO/ESTRUCTURA_PROYECTO/PIPELINE_PROYECTO en `docs/ESTRUCTURA.md`).
- Notebooks: 12 en total, numerados globalmente 01–09 por familia, sin redundancia
  aparente (cada número = una etapa distinta). No requieren consolidación.

## 3. Rutas hardcodeadas y configuración

Patrón consistente en `demo/*.py`:
- `REPO = ~/Desktop/labios-argentos` (demo_web, demo_ptt, demo_stream, build_testset,
  score_selftest) — **rompe si el repo se clona en otro lado** (p. ej. `~/Documents`).
- `VISPER_PY = ~/miniconda3/envs/visper/bin/python` (ídem).
- `infer_server.py`: `REPO = ~/Desktop/visper` (repo externo ViSpeR).
- `score_selftest.py`: además `~/Desktop/Visual_Speech_Recognition_for_Multiple_Languages`
  y checkpoint `modelos/ft05_espnet1.pth` con path absoluto.
- Los knobs de runtime YA usan env vars (`VSR_QWEN`, `VSR_BEAM`, `VSR_QMODEL`) —
  el patrón para paths es extenderlo, no inventar otro sistema.
- Scripts de GCP (`vsr/historical/ronda2/scripts/*.sh`, `personalization/calibracion/*.sh`)
  tienen paths de VM y bucket: son templates de startup, se documentan pero no se tocan.

## 4. Referencias a buckets (3 buckets distintos)

| Bucket | Refs | Propósito inferido |
|---|---|---|
| `gs://labios-argentos-vsr-dataset` | 20 | pesos + dataset de entrenamiento (AGENTS.md lo nombra canónico) |
| `gs://labios-argentos-vsr-data` | 14 | datos de la fase full-clean-release |
| `gs://labios-argentos-vsr-clean-v1` | 3 | release limpio v1 (rama `feature/full-clean-release`) |

No hay credenciales ni secretos versionados (verificado por grep; `.env`/keys ausentes).

## 5. Branches y PRs con material único

| Rama | vs main | Material único |
|---|---|---|
| `chore/reorganizacion` | +0 | — (mergeada, PR #27) |
| `feature/visual-audit-eval-prep` | +0 | — (mergeada, PR #24) |
| `fix/llm-correction-review` | +0 | — (mergeada) |
| `realtime/demo-kiosko` | +0 | — (mergeada) |
| `vsr/bigger-finetuning` | +0 | — (mergeada, PR #26 vía calibracion) |
| `feature/data-discovery-v1` | +3 | contenida al 100% en full-clean-release |
| `feature/clean-bucket-v1` | +4 | contenida al 100% en full-clean-release |
| `feature/full-clean-release` (PR #25 abierta) | +23/−12 | **SÍ — ver abajo** |

### feature/full-clean-release (PR #25) — análisis

Diff vs main: 29.320 archivos, de los cuales **~29.000 son BORRADOS masivos** de
`data/clips/` y `dataset/` (la política "datos pesados al bucket"). Esa eliminación es
una **decisión de datos separada** que esta limpieza NO reproduce (regla explícita).

Material único portable (agregados, todos texto):

| Dir | Peso | Contenido |
|---|---|---|
| `data_pipeline/discovery/` | 0.8 MB | pipeline de búsqueda/score de fuentes nuevas (src+tests+outputs+60 JSON metadata) |
| `data_pipeline/release/` | 85 MB ⚠️ | manifests del release limpio, 30+ reportes md, 14 scripts |
| `cleaning/gpt_clean_v1/` | 6 MB | limpieza GPT de transcripciones (src+prompts+reportes+patch_log 5.3 MB) |
| `data_pipeline/inventory/` | 11 KB | inventario del bucket |
| raíz | — | 6 docs (2 duplicados de reports/, 2 planes de branch-workflow, REPO_MAP superseded) |
| `.gitignore` | — | bloquea `*.mp4` etc. — **contradice la política de main**, no portable tal cual |
| `requirements.txt` | — | + `faster-whisper` (lo usa data_pipeline/discovery) — portable |

⚠️ `data_pipeline/release/` NO se porta entero: 10 manifests CSV >1 MB suman ~82 MB. Esos
permanecen accesibles vía el **tag `dataset-clean-v1`** (= `7221b55`, penúltimo commit
de la rama) y el bucket. Se porta todo lo <1 MB (reports, scripts, manifests chicos).

### Tag `dataset-clean-v1`

Existe y apunta a `7221b55` ("Aplicar GPT manual y documentar estructura del dataset"),
**dentro de la rama de PR #25, NO en main**. Es la referencia estable del release limpio.
Mientras el tag exista, todo el material de esa rama es recuperable aunque la rama se borre.

## 6. Fuentes de verdad contradictorias

1. **Política de datos** (la contradicción grande):
   - main / `docs/ESTRUCTURA.md`: "clips alineados SÍ se versionan (son el dataset)".
   - `feature/full-clean-release`: borra clips+dataset del repo, `.gitignore` con `*.mp4`,
     "los datos pesados viven en el bucket".
   - → Se documenta en `docs/DATA_AND_ARTIFACTS.md` y queda como **decisión abierta**
     en NEXT_STEPS. Esta branch no cambia la política de main.
2. **Resultados**: `experiments/README.md` (tabla maestra) y `docs/RESULTS.md` (ledger)
   coexisten y los números spot-checkeados coinciden (ft05 65.05, ViSpeR zs 45.22,
   qwen n-best −3.04 significativo). Se declara **`docs/RESULTS.md` como ledger canónico**
   y `experiments/` como índice/narrativa (ya es la convención de AGENTS.md).
3. **Mapa del repo**: `REPO_MAP.md` (rama #25) describe la estructura PRE-reorganización
   → superseded por `docs/ESTRUCTURA.md`. No se porta.

## 7. Datos versionados vs externos

- **Versionado**: código, clips mp4+txt (`data/clips/`, `dataset/`), corpus, manifests
  (`data/metadata/`), splits congelados (`vsr/splits/`), docs, experimentos.
- **Solo local**: `data/videos/` crudos (regenerables), ROIs `.npz`, pesos `.pth`,
  grabaciones personales (`~/vsr_personal/`), modelos calibrados (`modelos/personal/`).
- **Bucket(s)**: pesos de modelos, dataset de release limpio, resultados de VMs.
- **Irreemplazable si se pierde el bucket**: pesos fine-tuneados (ft03–ft07, LoRAs)
  y el release clean-v1. Los clips de YouTube son re-descargables solo mientras los
  videos sigan online (semi-regenerables).

## 8. Riesgos de cada movimiento previsto

| Movimiento | Riesgo | Mitigación |
|---|---|---|
| Portar dirs de la rama #25 a paths originales | colisión de nombres | no hay: `data_pipeline/discovery/`, `data_pipeline/release/`, `cleaning/gpt_clean_v1/`, `data_pipeline/inventory/` no existen en main |
| No portar manifests >1 MB | pérdida de acceso | recuperables vía `git show dataset-clean-v1:<path>` + bucket; documentado |
| Env-var override de paths en demo | cambio de comportamiento | defaults idénticos a los actuales; `REPO` derivado de `__file__` es estrictamente más correcto (funciona desde cualquier clone) |
| Tocar `.gitignore` | des-versionar datos | NO se porta el `.gitignore` de la rama #25 |
| Nuevos docs | contradicción con existentes | se linkean al ledger, no copian números |

## 9. Archivos históricos pero útiles (se preservan)

- `vsr/historical/ronda2/` completo (corrida ronda 2, evidencia de ft03–ft07).
- `docs/archivo/HANDOFF_ROIS_FINETUNE.md`.
- `docs/PLAN_CURRICULUM.md` (plan pausado, con contexto).
- `docs/bibliografia/paper-3.pdf` (referencia bibliográfica).
- `TO-DO.md` (checklist de la rúbrica, actualizado hoy).
- Notebooks 01–09 (cada uno es evidencia de una etapa).

## 10. Archivos regenerables / no clasificados

- Regenerables (ya ignorados, no versionados): `.npz`, videos crudos, previews, runs.
- No se detectaron archivos temporales, cachés versionados ni carpetas vacías en main.
- Sin clasificar: ninguno pendiente — todos los módulos de main tienen propósito claro.

## 11. Validaciones de esta fase

- `python3 -m compileall` sobre todos los `.py` materializados: **exit 0** (compilan).
- grep de secretos (`.env`, tokens, keys): sin hallazgos.
- Carpetas vacías: ninguna.
- READMEs faltantes: `demo/`, `vsr/curriculum/`, `vsr/mpc001/`, `vsr/historical/ronda2/`,
  `docs/bibliografia/` (se agregan en esta limpieza).
