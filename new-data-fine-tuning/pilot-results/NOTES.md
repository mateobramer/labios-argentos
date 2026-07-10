# Piloto — resultado (2026-06-27)

## Estado: PARCIAL. Falta un `apt install` y re-correr el preproc.

### Qué funcionó ✅
- VM L4 creada, entornos OK, código en la VM.
- ETAPA 1 (descarga + transcripción **large-v3** + corte de clips): **las 4 fuentes, 902 clips crudos**.

| Fuente | Clips crudos |
|---|---|
| RESPONDO TODO (Pablo Agustín) | 365 |
| BULLRICH... (Pedro Rosemblat) | 192 |
| MI PRIMER AMOR (Marti Benza) | 187 |
| La PRIMERA VEZ... TEL0 (Tomás Mazza) | 158 |
| **TOTAL** | **902** |

### Qué falló ❌
- ETAPA 2 (preproc visual MediaPipe) crasheó **al iniciar los workers**:
  `OSError: libGLESv2.so.2: cannot open shared object file`.
- Causa: falta una lib de sistema que MediaPipe necesita para el FaceLandmarker (GLES/EGL).
  Ya teníamos `libgl1`/`libglib2.0-0` (arregló `cv2`), pero faltan **`libgles2` y `libegl1`**.
- Resultado: **0 npz**, sin keep/review/drop. **No tenemos el número de attrition todavía.**

### Fix (listo para aplicar al reanudar)
1. `gcloud compute instances start labios-vsr-gpu --zone=us-central1-a`
2. En la VM: `sudo apt-get install -y libgles2 libegl1 libegl-mesa0`
3. Re-correr SOLO el preproc + detección (la ETAPA 1 ya está hecha, los 902 clips persisten en disco):
   - `~/venv-visual/bin/python -m visual_preprocessing.src.preprocesar --jobs 4`  (bajo a 4 jobs por las dudas de RAM con 902 clips)
   - `~/venv-visual/bin/python -m data_cleaning.src.detectar_clips_malos`
4. Mirar: crudos→npz (% supervivencia), keep/review/drop, y spot-check de alineación.

### Costo
- VM corrió ~41 min (21:49→22:30Z) → **~$0.58**. Detenida desde 22:30Z (solo disco ~$0.3/día).
- El re-run del preproc estimado: ~15–25 min → ~$0.3–0.5.

### UPDATE 23:13Z — re-run bloqueado por capacidad (VM sigue apagada, $0 extra)
- Intenté reanudar la VM para aplicar el fix: **`ZONE_RESOURCE_POOL_EXHAUSTED`** — no hay
  L4 on-demand en us-central1-a ahora. El disco es zonal → la VM solo arranca en esa zona.
- El job autónomo se colgó ~22 min reintentando SSH contra una VM que nunca prendió, y el
  harness lo mató antes del stop. **Nunca prendió → no gastó nada.** VM TERMINATED.
- **Insight clave:** el preproc (MediaPipe) corre en **CPU**; la L4 solo hace falta para
  Whisper (ya hecho) y el training (después). Entonces el re-run del preproc se puede hacer
  en una **VM CPU barata (sin stockout)**, no hace falta esperar capacidad de L4.

### Caminos para retomar (cuando estés / des OK)
1. **VM CPU desde snapshot del disco** (recomendado): snapshot del disco de `labios-vsr-gpu`
   → crear VM CPU (ej. e2-standard-8, ~$0.27/h, con `--max-run-duration` + auto-delete por
   seguridad de costo) en cualquier zona → `apt install libgles2 libegl1` → correr preproc
   `--jobs` + detección → bajar resultados → borrar. No depende de capacidad de L4.
2. **Reintentar L4**: volver a `instances start` cuando haya capacidad (intermitente).

### RESULTADO FINAL (28-06 ~01:00Z) — vía VM CPU desde snapshot ✅

| Fuente | Crudos | npz | Supervivencia |
|---|---|---|---|
| RESPONDO TODO (Pablo Agustín) | 365 | 354 | 97.0% |
| MI PRIMER AMOR (Marti Benza) | 187 | 164 | 87.7% |
| BULLRICH (Pedro Rosemblat) | 192 | 102 | 53.1% |
| La PRIMERA VEZ TEL0 (Tomás Mazza) | 158 | 29 | 18.4% |
| **TOTAL** | **902** | **649** | **72.0%** |

keep/review/drop de los 649: **649 keep / 0 review / 0 drop** → lo que pasa el preproc está limpio.

**Lecturas:**
- Promedio 72% supera el histórico (~60%) → talking-heads ayudan. PERO **varianza enorme**:
  Pablo 97% y Marti 88% (excelentes) vs Rosemblat 53% (el micrófono cruzando el mentón, como
  se predijo) y **Mazza 18%** (sorpresa: casi inusable; la frontalidad por heurística de formato
  NO siempre se cumple — el gate visual es el árbitro real).
- Implicancia para las 39: NO asumir 72% parejo. Algunas fuentes van a rendir casi nada. De
  10.7h crudas, el rinde curado es incierto (si hay varias tipo Mazza, baja).
- **Alineación NO verificada todavía** (salté `auditar_alineacion` en el piloto). Riesgo bajo
  por ser hablante único, pero conviene correrlo en la tanda completa.

**Riesgo para la fase siguiente:** la tanda completa (Whisper de 39 videos) y el entrenamiento
**SÍ necesitan L4**. Hoy hay stockout intermitente de L4 on-demand en us-central1-a → puede
bloquear/demorar. A decidir: reintentar L4, o pedir aumento de cuota/otra región.

**Costo total piloto:** ~$0.63 (L4 ~$0.58 + CPU ~$0.05). VM L4 sigue STOPPED (disco ~$0.33/día).

### AUDITORÍA DE ALINEACIÓN (28-06 ~16:30Z) — modelo `base`, 902 clips ✅

| Fuente | ok | drift | texto_dudoso |
|---|---|---|---|
| RESPONDO TODO (Pablo Agustín) | 341 | 0 | 24 |
| MI PRIMER AMOR (Marti Benza) | 117 | 4 | 66 |
| La PRIMERA VEZ TEL0 (Mazza) | 118 | 3 | 37 |
| BULLRICH (Rosemblat) | 146 | 3 | 43 |
| **TOTAL** | **722 (80%)** | **10 (1.1%)** | **170 (19%)** |

- **Drift 1.1% → alineación LIMPIA** (sin patrón sistemático). Riesgo de desfase/multi-hablante: descartado para estas fuentes.
- `texto_dudoso` 19% **inflado por usar `base`** en la auditoría (escucha peor → falsos dudosos). Los `.txt` reales son large-v3. Señal confiable = drift (nulo).
- Muestra de 25 clips contiguos/fuente en `sample-clips/` para ver a ojo.

### Lección operativa (para la fase de training)
- **Las subidas a GCS desde la VM fallaban** (permiso del service account sobre el bucket). Se perdió 1 corrida del audit por confiar solo en GCS. **Usar `scp` directo** para traer artefactos; arreglar el permiso del SA antes de subir checkpoints a `gs://labios-argentos-vsr-data/`.

### Costo total piloto: ~$1.0 (L4 ~$0.58 + 3 corridas CPU). Nada quedó prendido.

### Artefactos en esta carpeta
- `pilot.log` — log completo del piloto.
- `lip_preprocessing_manifest.csv` / `auditoria_clips_manifest.csv` — manifests del dataset
  PREVIO (no tienen filas nuevas porque el preproc no llegó a escribir).
- Los 902 clips nuevos quedaron en el disco de la VM (`~/labios-argentos/data/clips/`).
