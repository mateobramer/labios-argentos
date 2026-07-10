# TO-DO — empaquetado de producto (rúbrica Engineering Project)

Contexto: la profundidad ML ya está (ver `experiments/`); lo que falta puntúa en la rúbrica
(Arquitectura/Deployment/Robustez/Documentación). Todo es local, sin GPU.
**Actualizado 2026-07-10** con la reorganización del repo (bloques 1, 2 y 6 casi completos).

## 1. README.md de producto (Documentación + pitch) — ✅ hecho en la reorganización
- [x] Problema real que resuelve (accesibilidad / entornos ruidosos / subtítulos en vivo).
- [x] Diagrama de arquitectura (captura → landmarks → VAD visual → crop mean-face → encoder MPS →
      beam CPU → rescorer qwen → UI web).
- [x] Quickstart (3 comandos).
- [ ] Screenshot de la UI web + GIF/video corto de la demo (falta grabarlos).
- [x] Tabla resumen de resultados (test-658 + self-test 100) con link a `experiments/`.

## 2. docs/SPEC.md — especificación del sistema — ✅ hecho en la reorganización
- [x] Componentes, responsabilidades, interfaces (protocolo stdin/stdout del infer_server, CONFIG/READY).
- [x] Flujo de datos end-to-end con latencias medidas por etapa.
- [x] Cada decisión de diseño justificada con su experimento (beam=3, encoder-MPS, qwen top-5/4b,
      VAD por pausas, CPU vs int8). Es nuestro diferencial: ninguna decisión es "porque sí".
- [x] Limitaciones honestas (offline/bidireccional, WER por condición, un solo hablante validado).

## 3. Reproducibilidad / Deployment
- [ ] `environment.yml` (o requirements) por env: `ptt`, `visper` (+ `mvsr` opcional para ft05).
- [ ] Paths hardcodeados (`~/Desktop/visper`, `~/Desktop/labios-argentos`, binarios conda) → config
      (env vars o `config.yaml` chico).
- [ ] `setup.sh` (crea envs, baja face_landmarker.task, verifica visper_vsr_base.pth) + `run.sh`
      (levanta demo web con un comando).
- [ ] Nota honesta de Docker: MPS no existe dentro de contenedores en macOS → documentar variante
      CPU-only (más lenta) y que el deployment evaluable es la **demo en vivo** local.
- [ ] Instrucciones de pesos: dónde conseguir visper_vsr_base.pth (1.1 GB, no va al repo) — hoy solo
      mencionado en el README; falta el paso a paso.

## 4. Robustez + tests
- [ ] `tests/` con pytest: `norm()`/WER/CER (casos ñ/tildes), dedup de bordes del stitching,
      máquina de estados del VAD (serie sintética de apertura), parseo de manifest, protocolo CONFIG/READY.
- [ ] infer_server caído → reintento/restart automático en demo_web (hoy: error por línea).
- [ ] Mensajes claros: cámara sin permiso. (Ollama apagado ya cae a 1-best ✓ y quedó documentado en SPEC §6.)
- [x] Sección "manejo de errores y edge cases" en el SPEC.

## 5. Eficiencia y Costo
- [x] Sección de costos en README + SPEC §7 (todo local $0, GCP spot, calibración ~$0.05).
- [ ] qwen local vs API: comparación costo/latencia/privacidad con números.
- [ ] Tabla de configs velocidad vs WER en el SPEC (hoy linkea a experiments/09; evaluar copiarla).

## 6. Extras vistosos (si sobra tiempo)
- [ ] Confianza en la UI: colorear caption según score del beam (~30 min).
- [ ] Grabar 1-2 familiares (~30 clips c/u) → validación de generalización de hablante (el hueco
      de validación real que queda). `build_testset.py` ya es append+resumible.
- [x] Repo limpio en GitHub: reorganización 2026-07 (rama `chore/reorganizacion`): docs raíz viejos
      fusionados en `docs/ESTRUCTURA.md`, README/AGENTS reescritos, `realtime/` retirado,
      `.gitignore` al día, demo/ + experiments/ versionados.

---
*Generado 2026-07-09, actualizado 2026-07-10. Los experimentos de ciencia adicionales se coordinan aparte.*
