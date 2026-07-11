# 05 — Self-test limpio (usuario leyendo frases)

**Motivación:** el test-658 son clips scrapeados de YouTube (ángulos, oclusiones, calidad variable).
Pregunta: ¿parte del WER alto es el test set, o es el modelo? → grabar al usuario **de frente, de cerca,
buena luz**, leyendo frases conocidas (condición ideal), y medir.

**Herramientas:** `personalization/build_testset.py` (env `ptt`, MediaPipe crop, muestra frase → ESPACIO grabar/cortar
→ npz + manifest) y `personalization/score_selftest.py --model {ft05,visper}`. Datos en `~/vsr_selftest/`.
El usuario es hablante NUEVO (no visto en train) → sigue siendo speaker-independent (comparación justa).

## Ronda 1 — 12 frases

| Modelo | %WER | %CER | vs test-658 |
|---|---|---|---|
| **ViSpeR** | **37.29** | **19.29** | mucho mejor que 45/27 |
| ft05 (50M) | 67.80 | 46.55 | ~igual que siempre (65-68) |

2 frases exactas con ViSpeR ("no lo puedo creer se me rompio el celular de nuevo"; "no tengo ganas de
cocinar pidamos una pizza").

## Ronda 2 — 40 frases (más n, para estadística)

| Enfoque (ViSpeR) | %WER | %CER |
|---|---|---|
| 1-best | 23.60 ± 6.6 | 11.41 ± 3.5 |
| qwen n-best rescoring | 20.22 (−3.4) | 9.89 (−1.5) |
| oracle-5 | 16.57 | 8.32 |

(ViSpeR grabado aún más limpio esta ronda → CER 11. Detalle del rescoring en [04](04_qwen_corrector.md) §F.)

## Conclusiones

1. **El test set SÍ subestimaba a ViSpeR:** en limpio baja de 45/27 (test-658) a 23.6/11.4 (self-test).
   Los clips ruidosos de YouTube penalizaban al modelo bueno. En condición ideal, ViSpeR es **genuinamente
   usable**.
2. **ft05 NO mejora ni en limpio** (~68 WER): su techo es el modelo (pre-entreno chico), no el test.
   → confirma que para mejorar ft05 hace falta escala de datos, no limpiar el test ni post-procesar.
3. A **CER ~11** (ViSpeR limpio) el corrector LLM por fin ayuda **vía n-best rescoring** — es el régimen
   donde la hipótesis del CER se cumple. Con n=40 el IC todavía no cerraba significancia — resuelto
   en la Ronda 3 (abajo), ampliando a n=100.

## Ronda 3 — 100 frases (60 nuevas, para significancia)

Ampliamos a 100 clips (`build_testset.py` ahora appendea + es resumible). Los 60 nuevos resultaron **más
difíciles** (grabados de una sentada, menos marcados): fp32 1-best **23.60 (viejos 40) vs 33.73 (nuevos 60)
→ 29.51 (100)**. No es bug — los 40 originales reproducen 23.60 exacto. El set quedó más realista/variado.

Con n=100 e IC bootstrap **pareado**, la pregunta que arrastrábamos por fin cierra:

| ViSpeR beam40 (n=100) | %WER | IC |
|---|---|---|
| 1-best | 29.51 | ±4.4 |
| **qwen n-best rescoring** | **26.46** | ±4.5 |
| delta | **−3.04** | **IC95 pareado [+0.71, +5.53] → SIGNIFICATIVO ✅** |

→ **El n-best rescoring baja el WER de verdad** (no ruido). Detalle en [04](04_qwen_corrector.md) §F2.
El IC del WER a n=100 (~±4.4) es ~1.5× más ajustado que a n=40 (~±6.6), como predice √n.

La generalización a más hablantes queda como trabajo futuro (ver
[`FUTURE_WORK.md`](../FUTURE_WORK.md) §4); el barrido de n-best (top-10 / qwen3.5:9b) se
cerró en [09](09_velocidad_inferencia.md) §G.
