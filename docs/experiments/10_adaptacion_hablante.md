# 10 — Adaptación al hablante de ViSpeR (¿escala el hallazgo de la rama de calibración?)

**Contexto:** la rama `calibracion/adaptacion-hablante` (compañero) mostró que adaptar el **50M** (ft06)
con ~96 frases de una persona baja su WER personal 66→55 (**−11**, n=15) vía fine-tune continuado
(lr 1e-5). Pregunta: ¿escala eso a **ViSpeR (288M)**, que ya está ~25 pts mejor? ¿Y se compone con el
n-best rescoring de qwen? (2026-07-09/10)

## Setup

- **Datos:** los 100 clips del self-test de Fede → split **congelado y estratificado** (viejos/nuevos):
  **train 60 / val 10 / test 30** (12 viejos + 18 nuevos). Versionado en
  `vsr/splits/personal_fede/`. **El test-30 nunca se toca para entrenar** — es el benchmark
  personal de acá en más.
- **Brazos** (ambos desde `visper_vsr_base.pth`, augment, early stop paciencia 5, máx 20 épocas, L4):
  - **A: LoRA** (r16/α32 en atenciones, lr 1e-4) — la receta que ya sabíamos que no rompe.
  - **B: full-FT continuado lr 1e-5** — la réplica de la receta del compañero, traducida a ViSpeR.
- **Eval:** test-30 personal con **top-5 candidatas** (beam 40, para qwen local) + **test-658 completo**
  (olvido catastrófico). Deltas con **bootstrap pareado** por clip.
- **Infra:** VM L4 spot (imagen `labios-img-visper`), startup autónomo con auto-destrucción de VM+disco.
  Costo total: **~$1.2**. Sanity: el eval GPU del base reprodujo el baseline local **al dígito** (29.18).
  Artefactos en `gs://labios-argentos-vsr-dataset/adaptacion_fede/out/` (incl. `adapt_lora_best.pth`).

## Resultados (test-30 personal + test-658)

| modelo | test-30 1-best | CER | oracle-5 | + qwen n-best | test-658 |
|---|---|---|---|---|---|
| ViSpeR zero-shot | 29.18 | 14.99 | 20.23 | 25.29 | 45.22 ± 1.9 |
| **A: + LoRA personal** ⭐ | **24.51** | **10.76** | 17.90 | 24.51 | **44.54 ± 1.9** |
| B: + full-FT lr 1e-5 | 43.19 ❌ | — | — | — | **98.69 ☠️** |

Deltas pareados (n=30): adaptación LoRA **−4.67** WER, IC95 [−1.15, +10.42] → **dirección clara, no
significativo a n=30**. qwen sobre LoRA: **±0.00** [−3.56, +3.50].

## Hallazgos

1. **La adaptación personal SÍ funciona sobre ViSpeR (vía LoRA):** −4.7 WER / −4.2 CER personales con
   solo 60 clips y ~4 min de L4 (early stop ép. 19, curva sana). Replica la dirección del hallazgo del
   compañero, sobre un modelo que ya era 25 pts mejor. Con n=30 el IC pareado aún cruza 0 → falta n para
   declararlo (la del compañero, n=15, está aún más lejos de significancia).
2. **Cero olvido catastrófico con LoRA:** test-658 44.54 vs 45.22 — hasta una pizca mejor. El adapter
   personal no rompe el modelo general (se puede cargar en la demo sin miedo).
3. **La receta del compañero (full-FT lr 1e-5) NO escala a 288M: COLAPSO TOTAL.** test-658 se va a 98.7
   (destruido) y hasta lo personal empeora (43.2 > 29.2). **Trampa importante:** el val_loss bajaba
   monótono durante el colapso — la CE teacher-forced sobre 10 clips del mismo hablante no detecta que
   el decoding libre se rompe. En el 50M la receta anda; en el 288M hay que usar PEFT/LoRA sí o sí.
4. **Adaptación y rescoring qwen son (parcialmente) REDUNDANTES:** sobre el base, qwen daba −3.9; sobre
   el adaptado, **0.0**. Y base+qwen (25.29) ≈ LoRA+qwen (24.51) ≈ LoRA solo (24.51). Explotan el mismo
   error recuperable: una vez que la adaptación corrige la "boca" del hablante, no queda señal en las
   candidatas para el LLM. Punto valioso para la pregunta de research del umbral de CER (a CER ~10.8 el
   n-best ya no sumó — aunque a n=30 va con cautela; contrastar con el −3.0 a CER ~11.4/n=100 de [04](04_qwen_corrector.md) §F2).
5. Bonus operativo: la eval test-658 a beam 40 en L4 tarda ~110 min/brazo (dominó el runtime del run).
   Para la próxima: beam 5-10 en evals intermedias.

## Conclusión

**Adoptar el método del compañero pero con LoRA, no full-FT.** Para la demo personalizada: cargar
`adapt_lora_best.pth` (mejor personal Y mejor general). Para cerrarlo científicamente: grabar más clips
de test (o un segundo hablante adaptado) para llevar el −4.7 a significancia.

## Próximos

- [ ] Cablear el adapter a la demo (`VSR_CKPT`) y probarlo en vivo.
- [ ] Más test personal (o 2º hablante) → significancia del −4.7.
- [ ] Curva de nº de clips de train (¿30 alcanzan, como en el 50M?).
- [ ] Loop activo de 2 pasos del compañero (frases dirigidas a los errores de la persona).
