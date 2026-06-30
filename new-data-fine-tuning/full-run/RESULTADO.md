# Fase A — Procesamiento de las 39 fuentes nuevas (COMPLETA)

**Fecha:** 2026-06-29 · **VM:** `labios-vsr-full` (GCP L4, us-central1-a, on-demand) · **Modelo Whisper:** large-v2.

## Resultado

- **39 fuentes intentadas → 34 procesadas, 30 con npz útiles.**
- **3244 clips labiales curados (.npz)** — TOTAL 4276 clips → 3244 npz = **75.9%** entre las fuentes que rindieron.
- **102 clips de música/alucinación filtrados** automáticamente (`music_dropped.log`).
- Estos clips entran a **train** (val/test quedan congelados → comparación válida con v1/v2). Suma in-domain
  informal: roughly +60% sobre el train actual (~4818 clips de ft01).

### Rinde por tipo de fuente (destacados)
- **Muy bueno (>90%):** RESPONDO TODO/Pablo Agustín 98%, Cande Copello "bullying" 97%, ESTOY EN UN BROTE 94.5%,
  Rosemblat "MILITANDO" 94.5%, Marti Benza "estafaron" 93.5% / "balcón" 95%, Lucas Castel "me roban" 93%.
- **Medio (45-90%):** la mayoría de Marti Benza, Rosemblat, Mazza largo, Julian Serrano, paneles con La Reini.
- **Bajo/cero:** Lucho Mellera (era radio "Lucho con La Gente", 0), varios Mazza no-frontales (0-49%),
  GRWM de Sofía Gonet (28%), "LO MALO DE EE.UU" Spadafora (4.8%).

### Fuentes que fallaron la descarga (5) — recuperables
Casi seguro **bloqueo anti-bot de YouTube por IP** tras ~38 descargas (+ 1 age-gate):
- 21 Spadafora `aQlIHv_K0zk` (age-gate), 36 Connie Isla `ZhPgRWjBWvk`, 37/38/39 Martina tu bella amiga
  (`scQ7nPWsA8g`, `zyr7wpiIt18`, `wB4JpMNFqb4`).
- Se pueden reintentar más adelante con cookies/otra IP si vale la pena (son ~4-5 fuentes).

## Incidentes resueltos en el camino
1. **Crash de entorno** (libGLESv2) → resuelto instalando libs GL/EGL de entrada.
2. **Bug yt-dlp `--format mp4`** (videos sin mp4 progresivo) → parcheado a `bv*+ba/merge mp4` (recuperó la fuente 6).
3. **Bug CSV** (coma sin comillas en una nota corría columnas) → corregido, fuente 13 recuperada.
4. **Cuelgue de la VM** (~2h12 de run, sin OOM, probable driver GPU) → reset + resume reentrante (sin perder datos).
   Costó ~3h y ~$2.5 de VM tildada; el Monitor no lo cazó (dependía de SSH) → se le sumó alerta de no-respuesta.

## Artefactos
- **Datos curados (3244 npz + clips + manifests):** en el disco de `labios-vsr-full` (apagada) y en el
  **snapshot `labios-full-20260629-0429`** (40 GB, durable). De ahí sale el dataset para entrenar.
- **Local (`full-run/`):** `run2.log`, `lip_preprocessing_manifest.csv`, `auditoria_clips_manifest.csv`, `music_dropped.log`.

## Costo
- VM L4 ~11h17m → **~$9.59**. Total del proyecto (piloto incluido) ~**$11**. Presupuesto: $47.44.
- VM **TERMINATED**. Solo factura el disco (~$0.40/día) hasta que se borre o se use para entrenar.

## Próximos pasos
1. (Opcional) Recuperar las 5 fuentes caídas con cookies de YouTube.
2. **Re-armar splits** (`armar_splits` toma las nuevas a train; verificar speaker-independent).
3. **Entrenar ft03 (config v1) y ft04 (config v2)** sobre old+new → evaluar en test 658 → comparar vs v1/v2.
   - Necesita GPU otra vez (ojo stockout L4). Decidir antes: arreglar permiso GCS del SA para subir checkpoints.
