# Baseline zero-shot — VSR español (Gimeno, LIP-RTVE) sobre rioplatense

**Fecha:** 2026-06-22 · **VM:** `labios-vsr-gpu` (GCP, L4) · **Modelo:** `vsr-liprtve-si.pth`
(Conv3D-ResNet18 + Conformer + CTC/attention, ESPnet) del repo
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

1. **Desalineación clip↔texto en parte de nuestros datos.** En los clips finales de
   "ME ACUSARON" la hipótesis matchea el `ref` *vecino* (p. ej. la hyp de un clip coincide
   con el texto del clip siguiente) → el corte de Whisper desfasó texto y video. Es un QA
   pendiente del proyecto (ver `próximos pasos` del README) y sube el WER sin culpa del modelo.
2. **Cambio de dominio fuerte:** TV peninsular guionada → YouTube rioplatense espontáneo
   (voseo, léxico local, iluminación/encuadre variables).
3. **Subset chico (149 clips, 2 hablantes).** Suficiente para validar el pipeline y fijar un
   primer número; ampliar para un WER más estable.

## Reproducir

Ver `evaluation/README.md`. Resumen: `visual_preprocessing` → `evaluation.src.exportar_para_gimeno`
→ `gimeno_patches/aplicar_parches.py` → `vsr_main.py --database Rioplatense --scenario zero-shot`.

## Artefactos

- `liprtve-si_noLM/test.inf` — pares `ref#hyp` por clip · `test.wer` — WER/CER ± IC.
- `mapeo.csv` — sampleID → fuente/clip original (trazabilidad).
