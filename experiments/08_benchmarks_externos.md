# 08 — Benchmarks externos (contexto del estado del arte)

**Ojo:** cada número es sobre su PROPIO test set, NO sobre `test-658`. Sirven para ubicar nuestros
números, no para comparar 1:1. Todos offline/bidireccionales salvo aclaración.

| Sistema | Repo | Dataset (naturaleza) | Métrica | Datos train | Params |
|---|---|---|---|---|---|
| Auto-AVSR (VSR) | mpc001/auto_avsr | LRS2 (TV BBC, EN) | 14.6% WER | 3448h | 250M |
| Auto-AVSR (VSR) | mpc001/auto_avsr | **LRS3 (TED, EN)** | **19.1% WER** | 3448h | 250M |
| Auto-AVSR (VSR) | mpc001/auto_avsr | LRS3, solo 818h | 36.3% WER | 818h | 250M |
| mpc001 multilingüe | mpc001/VSR-multi | LRS3 (TED, EN) | **19.1% WER** | — | **~50M** |
| mpc001 multilingüe | mpc001/VSR-multi | CMLR (noticiero, ZH) | 8.0% CER | — | ~50M |
| mpc001 multilingüe | mpc001/VSR-multi | **CMU-MOSEAS ES** | **43.9-44.5% WER** | 16h | ~50M |
| Gimeno | evaluating-end2end-spanish | VLRF (lab, ES) | 24.8% WER (spk-dep) | 3h | ~102M |
| Gimeno | evaluating-end2end-spanish | **LIP-RTVE (RTVE, ES)** | **34.5% dep / 59.5% indep** | 13h | ~102M |
| PyTorch AV-ASR (**streaming**) | pytorch blog | LRS3 — **audio+video** | 1.6-2.6% WER | — | 35-383M |

**Chaplin** (`amanvirparhar/chaplin`): checkpoint Auto-AVSR LRS3 (19.1%) + MediaPipe + corrección Qwen3-4B.
NO es streaming: push-to-talk (grabar-soltar, inferencia offline). **Blueprint de nuestro demo.**

## Lecturas clave (por qué importan)

1. **El 50M NO está limitado por capacidad:** mpc001 (~50M) llegó a **19.1% en LRS3, igual que Auto-AVSR
   de 250M**. Con datos suficientes, el 50M compite con uno 5× más grande. → El problema de ft05 es DATOS,
   no tamaño.
2. **Datos y acento dominan:** LRS3 EN 19.1% → CMU-MOSEAS ES 44.5% → rioplatense zero-shot 71.5 (mpc001) /
   45.2 (ViSpeR). LRS3 sube de 19.1 a 36.3 al bajar de 3448h a 818h. **Nosotros tenemos ~19h rioplatenses.**
3. **Nuestro análogo real = LIP-RTVE speaker-independent (59.5% WER):** ft05 (65) está a ~5 pts, con menos
   datos y acento más lejano → resultado defendible.
4. **Nadie tiene VSR visual-only en streaming** (el único streaming real es audio-visual). Es hueco abierto.

Fuentes: Auto-AVSR arXiv:2303.14307 · mpc001 arXiv:2202.13084 · Gimeno arXiv:2502.00464 ·
ViSpeR arXiv:2406.00038 · Chaplin github.com/amanvirparhar/chaplin · PyTorch real-time AV-ASR blog.
