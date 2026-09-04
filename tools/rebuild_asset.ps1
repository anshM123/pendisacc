# Full asset rebuild. Each stage is its own process on purpose:
# URDF conversion and SimulationContext cannot coexist (access violation).
$ErrorActionPreference = "Continue"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$ROOT = "C:\Users\anshm\Downloads\pendisaac"
$PYI  = "$ROOT\env_isaaclab\python.exe"
$PYC  = "$ROOT\env_cad\Scripts\python.exe"

Write-Output "=== 0. conventions self-test ==="
& $PYI "$ROOT\dynamics\conventions.py"
if ($LASTEXITCODE -ne 0) { Write-Output "conventions FAILED"; exit 1 }

Write-Output "=== 1. URDF from CAD ==="
& $PYC "$ROOT\tools\build_urdf.py"
if ($LASTEXITCODE -ne 0) { Write-Output "urdf FAILED"; exit 1 }

Write-Output "=== 2. URDF -> USD (isolated process) ==="
& $PYI "$ROOT\tools\build_usd.py" *> "$ROOT\tools\usd_build_log.txt"
Write-Output "   build_usd exit=$LASTEXITCODE"

Write-Output "=== 3. validate articulation (fresh process) ==="
Remove-Item "$ROOT\results\asset_validation.json" -ErrorAction SilentlyContinue
& $PYI "$ROOT\tools\validate_asset.py" *> "$ROOT\tools\validate_log.txt"
Write-Output "   validate exit=$LASTEXITCODE"
Write-Output "=== DONE ==="
