# Baseline zero-shot — VSR español (Gimeno, LIP-RTVE) sobre rioplatense

**Fecha:** 2026-06-22 (actualizado 2026-06-23) · **VM:** `labios-vsr-gpu` (GCP, L4) ·
**Modelo:** `vsr-liprtve-si.pth` (Conv3D-ResNet18 + Conformer + CTC/attention, ESPnet) del repo
[`david-gimeno/evaluating-end2end-spanish-lipreading`](https://github.com/david-gimeno/evaluating-end2end-spanish-lipreading).

## Qué se midió

WER/CER de un modelo entrenado en **español peninsular** (LIP-RTVE, TV de RTVE, escenario
speaker-independent) evaluado **sin ningún fine-tuning** sobre **149 clips rioplatenses**
de YouTube (2 hablantes: storytime "LE DIJE QUE SOY ARGENTINO" 80 clips + "ME ACUSARON DE
BRUJA" 69 clips). ROIs labiales 96×96 gris a 25 fps (warp a cara media estilo Auto-AVSR),
texto limpiado `lower+unidecode+ñ`.

## Resultados

| Decoding | %WER | %CER | n |
|---|---|---|---|
| **sin LM** (titular) | **79.55 ± 2.49** | **49.57 ± 2.09** | 149 |
| con LM (`lm-liprtve`, lm_weight 0.4) | 80.21 ± 2.79 | 52.46 ± 2.24 | 149 |

**Referencia:** el mismo modelo da **59.5% WER** en su propio test (LIP-RTVE-si). La
degradación peninsular→rioplatense-YouTube es de ~20 puntos de WER. El CER ~50% muestra
que el modelo acierta cerca de la mitad de los caracteres: **está leyendo labios, no
adivinando.**

**El LM peninsular no ayuda — empeora levemente** (WER +0.7, CER +2.9). El modelo de
lenguaje entrenado en texto peninsular mete sesgo léxico que no calza con el rioplatense
(voseo, jerga local), así que en vez de corregir, desvía. Es evidencia directa a favor de
**un corrector propio adaptado al rioplatense** (el LLM local del proyecto) en lugar de un
LM de dominio ajeno.

## Actualización (2026-06-23): re-corrida sobre el dataset corregido

Tras regenerar el dataset eliminando el desfase clip↔texto (corte por palabra/pausa en
silencios reales + re-transcripción con Whisper `large-v3`; ver caveat 1), se repitió el
zero-shot **sin LM** sobre los **mismos 2 hablantes** con los clips corregidos.

| Decoding | %WER | %CER | n |
|---|---|---|---|
| sin LM — dataset original (con desfase) | 79.55 ± 2.49 | 49.57 ± 2.09 | 149 |
| **sin LM — dataset corregido** | **79.26 ± 2.42** | **47.20 ± 1.85** | 160 |

**Lectura:** corregir el desfase deja el **WER prácticamente igual** (−0.3 pts, dentro del
intervalo de confianza) y mejora el **CER ~2.4 pts**. Confirma la hipótesis del caveat 1: el
desfase inflaba el WER de forma **menor** (~1 pt), **no es la causa** de la brecha de ~20 pts
contra el peninsular — esa brecha es **cambio de dominio / acento rioplatense real**. El efecto
es chico además porque solo "ME ACUSARON" (f02) tenía desfase; "LE DIJE" (f01) ya estaba
alineado, y el promedio sobre 160 clips diluye la mejora en los clips afectados. Aun así, el
dataset corregido es lo correcto para el entrenamiento/destilación posterior, aunque el WER
zero-shot casi no baje. Artefactos: `liprtve-si_noLM_corregido/test.{inf,wer}`.

## Validación (gate cualitativo)

No reprodujimos el 59.5% de LIP-RTVE (el Zenodo trae *landmarks*, no los videos del corpus;
requeriría bajar LIP-RTVE aparte — diferido). En su lugar:

- El modelo carga, corre y produce **español coherente correlacionado con la referencia**.
  Ejemplos de match de contenido: `interesante→interesante`, `montreal→montal`,
  `espectacular→vetacular`, `aproximadamente…habitantes`, `preconcepcion→precocepcion`,
  `disfrutando`, `pueblo`/`argentina` recurrentes. Si el crop estuviera mal, saldría ruido.
- Nuestro preprocesamiento usa el **mean-face de Auto-AVSR idéntico** (`20words_mean_face.npy`)
  del que derivan estos checkpoints → alta compatibilidad de crop.

## Caveats (por qué el WER está algo inflado)

1. **[RESUELTO 2026-06-23] Desalineación clip↔texto en parte de nuestros datos.** En los clips
   finales de "ME ACUSARON" la hipótesis matcheaba el `ref` *vecino* (la hyp de un clip coincidía
   con el texto del clip siguiente) → el corte de Whisper por timestamps de segmento desfasó texto
   y video. **Corregido**: el dataset se regeneró cortando por palabra/pausa (ver "Actualización").
   Impacto medido en el WER: **menor** (−0.3 pts WER, −2.4 pts CER) — no era el driver de la brecha.
2. **Cambio de dominio fuerte:** TV peninsular guionada → YouTube rioplatense espontáneo
   (voseo, léxico local, iluminación/encuadre variables). **Este es el factor dominante.**
3. **Subset chico (149–160 clips, 2 hablantes).** Suficiente para validar el pipeline y fijar un
   primer número; ampliar para un WER más estable.

## Reproducir

Ver `evaluation/README.md`. Resumen: `preprocessing` → `evaluation.src.exportar_para_gimeno`
→ `gimeno_patches/aplicar_parches.py` → `vsr_main.py --database Rioplatense --scenario zero-shot`.

## Artefactos

- `liprtve-si_noLM/test.inf` — pares `ref#hyp` por clip · `test.wer` — WER/CER ± IC (dataset original).
- `liprtve-si_noLM_corregido/test.{inf,wer}` — ídem sobre el dataset corregido (2026-06-23).
- `mapeo.csv` — sampleID → fuente/clip original (trazabilidad).
