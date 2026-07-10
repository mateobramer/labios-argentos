# vsr/historical/ronda2/ — corrida histórica de la ronda 2 (ft03–ft07)

**Estado: histórico** (completada 2026-06/07). Registro de la campaña que procesó las
39 fuentes nuevas de la ronda 2 y entrenó ft03–ft07 en VMs L4 spot de GCP. Los
resultados consolidados viven en [`../../../docs/experiments/01`](../../../docs/experiments/01_finetunes_50M_gimeno.md)
y el ledger [`docs/RESULTS.md`](../../../docs/RESULTS.md); acá queda la evidencia de la corrida.

| Path | Qué es |
|---|---|
| `PLAN.md` / `PLAN_ENTRENAMIENTO.md` | planes de la campaña (datos + entrenamiento) |
| `ESTADO_ACTUAL.md` | handoff de estado al 2026-06-29 (snapshot histórico, no refleja el presente) |
| `full-run/` | manifests de auditoría, logs y `RESULTADO.md` de la corrida completa |
| `pilot-results/` | ídem del piloto previo |
| `scripts/` | orquestación de VMs (autograb de GPU spot), filtro de música, análisis WER por longitud |

Los scripts de VM (`autograb_*.sh`, `train_orchestrator.sh`) son templates de startup
de GCP con paths internos de VM — se preservan como documentación de la campaña.
