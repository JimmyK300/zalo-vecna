param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Pass -Python with a Python 3.11+ executable.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.\.venv\Scripts\python.exe' scripts/rebuild_suites.py
    if ($LASTEXITCODE -ne 0) { throw 'Check ../train before creating suite.' }
    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
} finally { Pop-Location }
