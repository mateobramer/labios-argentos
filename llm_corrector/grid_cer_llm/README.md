# grid_cer_llm — scripts de la grilla del umbral CER/LLM

Reproducen el experimento de [`docs/experiments/04_qwen_corrector.md`](../../docs/experiments/04_qwen_corrector.md) §G:
¿a qué CER (y en qué dominio) el rescoring con un LLM mejora las transcripciones de lectura de labios?

## Pipeline

1. **Candidatas top-5** (una vez por celda modelo×test):
   - `infer_cell.py` — ViSpeR fp32 (encoder MPS + beam 40 CPU) o int8 (qnnpack). Env `visper`.
   - `infer_cell_ft.py` — ft05/ft07 remapeados a espnet1, beam 30 + LM CMU (variante con-LM). Env `mvsr`.
   - `vsr_nbest.py` — ft05/ft07 con el pipeline oficial de Gimeno en GPU (espnet2, beam 30, sin LM).
     Corre dentro de su repo en la VM; requiere agregar la base al whitelist de `MyDataset`.
   - `remap_ft07.py` — deriva el rename espnet2→espnet1 por posición y se autovalida contra ft05.
   - `export_selftest.py` — exporta los selftests al layout de datos de Gimeno.
2. **Fase LLM + scoring**: `grid_score.py` — corre qwen (corrección 1-best y n-best rescoring,
   prompts fijos adentro) persistiendo en los mismos JSON, y computa WER/CER/oracle + IC95
   pareado por celda. Reanudable; `--solo <archivos>` para procesar celdas puntuales.
3. **Análisis**: `estratos.py` (efecto por estrato de CER-por-clip), `diversidad.py`
   (diversidad de candidatas por celda), `figura.py` (genera `umbral_cer_llm.png` en este dir).

## Datos

Los JSON por clip (ref + 5 candidatas + `corr` + `resc`) viven en `data/` — todo el scoring
es recomputable sin re-inferir. `data/grid_puntos.json` (tabla máquina: deltas, ICs, oracle
por celda) y `data/grid_prompts_y_ejemplos.json` (prompts exactos + 6 ejemplos por celda) los
regenera `grid_score.py`. Requisito para la fase LLM: Ollama local con
`qwen3:4b-instruct-2507-q4_K_M`.
