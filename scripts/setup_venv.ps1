# Creates the project virtualenv and installs dependencies inside it only.
# No global pip changes. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_venv.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "[setup] Creating virtualenv at $Venv"
    python -m venv $Venv
}

$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

Write-Host "[setup] Upgrading pip (inside venv)"
& $Py -m pip install --upgrade pip

Write-Host "[setup] Installing requirements (inside venv)"
& $Pip install -r (Join-Path $Root "requirements.txt")

Write-Host "[setup] Done. Activate with:  .\.venv\Scripts\Activate.ps1"
