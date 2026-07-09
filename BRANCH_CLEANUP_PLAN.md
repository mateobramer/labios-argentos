# Branch cleanup ejecutado

Fecha: 2026-07-09
Branch actual: `feature/full-clean-release`
Release base: `7221b55b5a5ec9648a8098c6c0fb3324fad7a48f`

## Verificacion

Comandos de lectura ejecutados:

```powershell
git branch --merged feature/full-clean-release
git tag --list dataset-clean-v1
git ls-remote --tags origin dataset-clean-v1
```

Resultado:

- `feature/clean-bucket-v1`, `feature/data-discovery-v1` y
  `feature/visual-audit-eval-prep` estaban mergeadas en `feature/full-clean-release`.
- `dataset-clean-v1` no existia local ni remoto antes de crearlo.

## Ramas locales borradas

```powershell
git branch -d feature/clean-bucket-v1 feature/data-discovery-v1 feature/visual-audit-eval-prep
```

Resultado:

- `feature/clean-bucket-v1` borrada localmente; apuntaba a `17e44ef65`.
- `feature/data-discovery-v1` borrada localmente; apuntaba a `93c575fa8`.
- `feature/visual-audit-eval-prep` borrada localmente; apuntaba a `2076da557`.

No se borraron ramas remotas.

## Tag de release

Tag creado localmente:

```powershell
git tag -a dataset-clean-v1 7221b55b5a5ec9648a8098c6c0fb3324fad7a48f -m "dataset clean v1"
```

El tag apunta al commit final del dataset, no al commit posterior de limpieza local.

Tag pusheado:

```powershell
git push origin dataset-clean-v1
```

Resultado:

- `dataset-clean-v1 -> dataset-clean-v1` creado en `origin`.

## Main

No se toco `main`.
No se hizo force push.
No se hizo merge.

Recomendacion vigente:

- Integrar `feature/full-clean-release` a `main` con PR o merge controlado.
- Resolver explicitamente que `origin/main` tenia commits propios respecto del release.
- Mantener `dataset-clean-v1` como referencia estable del release final.
