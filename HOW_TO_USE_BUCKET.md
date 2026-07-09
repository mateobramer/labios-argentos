# HOW TO USE BUCKET

Bucket final:

```text
gs://labios-argentos-vsr-clean-v1/
```

Guia canonica:

```text
data_release/reports/HOW_TO_USE_BUCKET.md
```

Manifest principal:

```text
gs://labios-argentos-vsr-clean-v1/manifests/final_release_manifest.csv
```

Para entrenar VSR, usar `data_release/final_train_manifest_clean_gpt_v1.csv` y filtrar:

- `usable_for_training=true`
- `npz_path` no vacio

El texto recomendado esta en `selected_training_text`.
