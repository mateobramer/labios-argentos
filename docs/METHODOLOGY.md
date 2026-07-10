# Methodology — protocolo de evaluación

Reglas que hacen comparables todos los números del proyecto. El ledger de resultados es
[`RESULTS.md`](RESULTS.md); este doc define **cómo** se miden.

## Test set y splits congelados

- **`test-658`**: 658 clips de YouTube, **2 hablantes held-out** (f15/f22) nunca vistos
  en train → evaluación **speaker-independent**.
- Los splits val/test están **congelados** desde ft03 (`vsr/splits/`) y son los mismos en
  toda la serie ft03→ft07, así que las comparaciones son head-to-head.
- **Una comparación solo vale si usa exactamente estos splits.** No se modifican.
- **`self-test`**: clips propios grabados en condiciones controladas (buena luz, boca
  frontal). Mide el techo del sistema, no generaliza a YouTube variado.

## Normalización de texto (`norm()`)

Idéntica en todas las evaluaciones: minúsculas, sin acentos (**la ñ se preserva**), sin
puntuación, espacios colapsados. Implementación canónica: `vsr/mpc001/scripts/zeroshot.py`
(y replicada en `personalization/score_selftest.py`). WER y CER se calculan sobre este
texto normalizado.

## Métricas e intervalos de confianza

- **%WER** y **%CER** con **IC 95%** por bootstrap (2000 iteraciones).
- IC que no se solapan ⇒ diferencia estadísticamente significativa.
- Para comparar dos sistemas sobre los mismos clips (p. ej. 1-best vs n-best rescoring),
  se usa **bootstrap pareado** del delta: si el IC95 del delta excluye 0, el efecto es
  significativo. Es lo que confirma el −3.04 WER del rescoring a n=100
  ([exp. 04](experiments/04_qwen_corrector.md) §F2).

## Advertencias metodológicas registradas

- **Tamaño de muestra**: con n=12 la corrección 1-best dio un falso positivo (−1.7 WER)
  que n=40 desmintió. Los resultados sobre n=24 (prompts, few-shot) son orientativos, no
  concluyentes.
- **Mismas fuentes ≠ mismo N**: un número sobre 149 clips no es comparable a uno sobre
  658 aunque sean las mismas 2 fuentes (ver la corrección registrada en la corrida
  histórica ronda-2).
- **No se ajustan números para hacerlos consistentes.** Si dos documentos se contradicen,
  se investiga el origen y se documenta la contradicción; no se "emparejan" a mano.

## Arquitectura de los modelos propios (contexto)

Conv3D + ResNet18 → Conformer (12 capas) → decoder híbrido CTC/Attention.
Offline/bidireccional. ViSpeR = misma familia, 288M (adim768). Ver
[`RESULTS.md`](RESULTS.md) §1.
