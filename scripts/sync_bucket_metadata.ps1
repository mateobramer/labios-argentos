# Wrapper de Windows para scripts/sync_bucket_metadata.py.
# La logica real vive en el script Python; esto solo lo invoca con el interprete
# disponible y reenvia los argumentos tal cual.
#
# Uso:
#   .\scripts\sync_bucket_metadata.ps1
#   .\scripts\sync_bucket_metadata.ps1 --dry-run

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ScriptPath = Join-Path $PSScriptRoot "sync_bucket_metadata.py"

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    $Python = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $Python) {
    Write-Error "No se encontro 'python' ni 'python3' en PATH. Instalar Python 3."
    exit 1
}

& $Python.Source $ScriptPath @args
exit $LASTEXITCODE
