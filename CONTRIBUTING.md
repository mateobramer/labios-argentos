# Contributing

Guía para contribuir al proyecto. Instalación y comandos: [`docs/SETUP.md`](docs/SETUP.md).

## Convenciones de código

- Cada componente del sistema vive en su carpeta, con lógica reutilizable en `src/` y su
  `README.md`. No dejar scripts, datos ni resultados sueltos en la raíz.
- Código y comentarios en **español**; nombres de funciones en español
  (`bajar_video`, `cortar_clips`). Estilo procedural directo.
- Los directorios bajo `data/` son datos generados: no editarlos a mano.

## Dónde está la verdad (antes de re-medir algo, fijate si ya está medido)

- **Resultados**: [`docs/RESULTS.md`](docs/RESULTS.md) es el **ledger canónico** de
  métricas. [`docs/experiments/README.md`](docs/experiments/README.md) es el índice
  narrativo de los experimentos (no duplica números: enlaza al ledger).
- **Decisiones de diseño**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) y
  [`docs/SPEC.md`](docs/SPEC.md) — cada parámetro está justificado con su experimento.
- **Protocolo de evaluación**: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — splits
  congelados (`vsr/splits/`, test-658/val fijos desde ft03) y normalización de texto.

## Invariante sagrada del dataset

La **alineación video↔texto**: si tocás el corte de clips o el preprocessing, el `.txt`
debe seguir correspondiendo exacto al clip. Los splits congelados no se modifican — una
comparación solo vale con exactamente esos splits y la misma `norm()`
([METHODOLOGY](docs/METHODOLOGY.md)).

## Reglas de datos y costos

- **Privacidad**: grabaciones personales (`~/vsr_personal/`, `~/vsr_contrib/`) y modelos
  calibrados (`modelos/personal/`) **no se versionan nunca**.
- **No versionar pesados/regenerables**: `.npz`, `.pth`, videos crudos, venvs, clones de
  repos externos (ya en `.gitignore`; revisarlo antes de commits grandes). Política
  completa en [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md).
- **Recolección desde YouTube**: sin cookies del browser ni rotación de IPs; el scraping
  masivo quedó descartado (ver [experimento de datos](docs/experiments/07_datos_y_scraping.md)).
  Fuentes nuevas de a una, con los gates de calidad.
- **Costos en GCP**: las VMs de entrenamiento se lanzan spot, con auto-destrucción, y se
  verifica que murieron (VM y disco). No dejar nada corriendo sin monitor.

## Experimentos nuevos

Documentarlos en [`docs/experiments/`](docs/experiments/) (un doc por categoría) y
actualizar la tabla maestra en [`docs/RESULTS.md`](docs/RESULTS.md). El código del
experimento va dentro del componente correspondiente, no suelto.

## Nota sobre el working copy

El repo usa clones parciales/`sparse-checkout` para no materializar datos pesados. Si un
path trackeado "no existe" en tu working copy, revisá `git sparse-checkout list`. Los
datos masivos viven en buckets/tags — cómo recuperarlos: [`data/README.md`](data/README.md).
