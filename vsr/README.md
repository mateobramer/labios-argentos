# vsr

Módulo de entrenamiento, evaluación y contratos de datos para reconocimiento visual del
habla en español rioplatense. La evidencia numérica canónica vive en
[`docs/RESULTS.md`](../docs/RESULTS.md) y los experimentos en
[`docs/experiments/`](../docs/experiments/).

## Estado actual

| Modelo / corrida | Datos | %WER `test-658` | Estado |
|---|---|---:|---|
| ft05 (Gimeno/LIP-RTVE) | ~19 h rioplatenses | 65.05 | mejor fine-tune propio de esa familia |
| ViSpeR zero-shot | 794 h de pre-entrenamiento español/multilingüe | **45.22** | base general recomendada |
| ViSpeR full-FT argentino | +~19 h argentinas | 61.51 | degradó por sobreajuste |
| ViSpeR LoRA+augment argentino | +~19 h argentinas | 45.97 | empate técnico con zero-shot |

La adaptación **personal** por hablante vive en [`personalization/`](../personalization/):
LoRA mostró −4.67 WER personal en n=30, todavía sin significancia estadística, y no
degradó el test general. No confundir esa calibración personal con el LoRA global argentino.

## Contrato de datos

`vsr/splits/` contiene los splits speaker-independent congelados. Cada fila apunta a un
ROI `.npz` `(T, 96, 96)` uint8 gris a 25 fps y a su transcripción normalizada.

```text
vsr/splits/splits.csv
vsr/splits/{train,val,test}.csv
# columnas: split, spk, titulo, clip, n_frames, texto, npz
```

El test canónico es `test-658`: 658 clips de dos hablantes held-out. No modificar los
splits históricos para comparar una corrida nueva; crear un escenario o split separado.

## Estructura

| Path | Contenido |
|---|---|
| `src/` | armado de splits y fine-tuning de la familia Gimeno |
| `splits/` | train/val/test congelados y splits personales versionables |
| `evaluation/` | exportación, inferencia y métricas WER/CER |
| `historical/ronda2/` | scripts y evidencia de ft03–ft07 |
| `curriculum/` | preparación experimental de ViSpeR-es |
| `mpc001/` | notas/scripts del modelo multilingüe externo; el clon no se versiona |

## Comandos frecuentes

```bash
# Re-armar splits solo cuando el experimento lo requiera explícitamente
python -m vsr.src.armar_splits

# Evaluación y utilidades
python -m compileall -q vsr

# Demo con la base ViSpeR
bash run.sh
```

Los entrenamientos grandes corren en VMs L4 y los pesos no se versionan. Ubicaciones,
recuperación y buckets: [`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md).
La demo usa ViSpeR desde un clon externo indicado por `VISPER_DIR`; instalación:
[`docs/SETUP.md`](../docs/SETUP.md).

## Decisiones vigentes

- ViSpeR zero-shot es la base general por defecto.
- Full fine-tuning global con el dataset argentino actual no se recomienda: degradó el modelo.
- LoRA global evitó la degradación, pero no mejoró significativamente el WER general.
- LoRA personal es una línea distinta y prometedora; requiere más hablantes/test para cerrar significancia.
- La corrección LLM 1-best no se usa; el corrector opcional es n-best rescoring.
