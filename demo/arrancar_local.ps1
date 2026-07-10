<#
Arranca la demo completa en Windows con los recursos ya descargados localmente.
Las tomas se guardan fuera del repo para que sobrevivan cambios de codigo.
Uso: .\demo\arrancar_local.ps1
#>
param([int]$Port = 8551)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$base = Join-Path $env:USERPROFILE 'vsr-demo'
$python = Join-Path $base 'env\Scripts\python.exe'
$visper = Join-Path $base 'visper'
$capturas = Join-Path $base 'capturas'

foreach ($ruta in @($python, $visper)) {
  if (-not (Test-Path $ruta)) { throw "Falta $ruta. Revisá la instalación local de la demo." }
}
New-Item -ItemType Directory -Force -Path $capturas | Out-Null
$env:LABIOS_REPO = $repo
$env:VISPER_PY = $python
$env:VISPER_DIR = $visper
$env:VSR_PERSONAL_DIR = $capturas
$env:VSR_PROMPTS_FILE = Join-Path $env:USERPROFILE 'Downloads\prompts_vsr_rioplatense_1100_limpio.py'
$env:PYTHONUTF8 = '1'

& $python (Join-Path $repo 'demo\demo_web.py') --port $Port
