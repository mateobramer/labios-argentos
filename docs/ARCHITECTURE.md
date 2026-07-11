# Architecture — pipeline end-to-end y sistema de la demo

Visión de sistema. El detalle de la demo (interfaces, protocolo, parámetros) está en
[`SPEC.md`](SPEC.md); las decisiones justificadas por experimento, en
[`RESULTS.md`](RESULTS.md) y [`experiments/`](experiments/).

## El pipeline como componentes (01 → 07)

Cada etapa vive en su propia carpeta importable. La numeración es narrativa (cuenta el
flujo); los nombres en disco son paquetes Python válidos.

```
1. data_pipeline/    fuentes curadas → descarga+corte (yt-dlp+Whisper+ffmpeg) → clips mp4+txt
                     + discovery de fuentes, manifests, release limpio, inventario de bucket
        ▼
2. cleaning/         control de calidad: limpieza visual (keep/review/drop), limpieza GPT
                     de transcripciones, re-segmentado oracional
        ▼
3. preprocessing/    MediaPipe FaceLandmarker → 4 puntos → warp mean-face → ROI boca 96×96
        ▼
4. vsr/              modelos VSR (50M propio + ViSpeR 288M), fine-tuning, splits congelados,
                     evaluación WER/CER contra test-658
        ▼
5. llm_corrector/    n-best rescoring con LLM local (y los negativos: 1-best, prompts, few-shot)
        ▼
6. personalization/  calibración por hablante: grabación → splits personales → LoRA → eval
        ▼
7. demo/             app integrada en vivo: cámara → VAD visual → preprocessing online →
                     inferencia → rescoring opcional → subtítulos + feedback editable
```

## Arquitectura de la demo (tiempo de inferencia)

```
cámara (30 fps) ─────────────────────────────▶ MJPEG a la UI
   │
   ▼
MediaPipe FaceLandmarker (hasta 3 caras; sticky-lock a la más cercana)
   │  apertura de labios
   ▼
VAD visual (auto-calibrado; corta por pausa 0.45 s / tope 4 s)
   │  segmento de habla
   ▼
crop de boca 96×96 (warp mean-face) → .npz
   │
   ▼
infer_server (env visper — ViSpeR 288M, conformer)
   ├─ encoder ──── MPS   (~0.17 s)
   └─ beam search ─ CPU  (beam 3, ~0.9 s)
   │
   ▼
[opcional] qwen3:4b n-best rescoring (Ollama local, +1.2 s, −3.0 WER)
   │
   ▼
UI web (SSE): subtítulo + tira de boca + guion acumulado + corrección editable
```

**Latencia total: ~1.1 s por segmento (~2.3 s con el corrector)** en una MacBook M1.
Interfaces (protocolo `CONFIG`/`READY` del infer_server, endpoints HTTP): [`SPEC.md`](SPEC.md).

## Decisiones de diseño (cada una con su experimento)

| Decisión | Por qué | Evidencia |
|---|---|---|
| Modelo base ViSpeR 288M zero-shot | +20 WER vs el mejor fine-tune propio | [exp. 02](experiments/02_zeroshot.md) |
| beam = 3 | 2.2× más rápido, mismo WER | [exp. 09](experiments/09_velocidad_inferencia.md) |
| encoder en MPS, beam en CPU | frontend 3.4×; espnet rompe con el decoder en MPS | [exp. 09](experiments/09_velocidad_inferencia.md) |
| Corrector = n-best rescoring (no 1-best) | 1-best empeoró en todas las condiciones evaluadas; rescoring −3.04 WER (sig.) | [exp. 04](experiments/04_qwen_corrector.md) |
| Personalización = LoRA (no full-FT) | full-FT degradó severamente el 288M | [exp. 10](experiments/10_adaptacion_hablante.md) |
| VAD visual por pausas | no hay audio; ventana fija corta palabras | [exp. 06](experiments/06_demo_y_remap.md) |

Tabla completa en [`SPEC.md`](SPEC.md) §5. Limitaciones honestas: [`LIMITATIONS.md`](LIMITATIONS.md).
