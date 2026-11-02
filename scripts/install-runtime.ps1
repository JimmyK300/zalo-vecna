param([string]$Tag = 'b11960')
$ErrorActionPreference = 'Stop'
if ($Tag -notmatch '^b\d+$') { throw 'Tag must be an official nightly build number, e.g. b11960.' }
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runtime = Join-Path $repo 'tools\llama.cpp'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$assets = @("llama-$Tag-bin-win-cuda-13.4-x64.zip", 'cudart-llama-bin-win-cuda-13.4-x64.zip')
$saved = @()
foreach ($asset in $assets) {
    $destination = Join-Path $runtime $asset
    $partial = $destination + '.part'
    $url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/$asset"
    if (Test-Path -LiteralPath $destination) {
        & $python -c "import sys,zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" $destination
        if ($LASTEXITCODE -ne 0) {
            if (Test-Path -LiteralPath $partial) { throw 'Both incomplete ZIP and partial exist; inspect before retrying.' }
            Move-Item -LiteralPath $destination -Destination $partial
        }
    }
    if (-not (Test-Path -LiteralPath $destination)) {
        Write-Host "Downloading official runtime: $asset"
        & curl.exe --location --fail --silent --show-error --connect-timeout 20 --max-time 600 --retry 3 --retry-all-errors --speed-time 45 --speed-limit 1024 --continue-at - --output $partial $url
        if ($LASTEXITCODE -ne 0) { throw 'Runtime download failed; partial retained for resume.' }
        Move-Item -LiteralPath $partial -Destination $destination
    }
    Expand-Archive -LiteralPath $destination -DestinationPath $runtime -Force
    $saved += [PSCustomObject]@{file=$asset; source=$url; sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash}
}
[PSCustomObject]@{tag=$Tag; assets=$saved; note='Locally computed download hashes; official GitHub release origin.'} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runtime 'runtime_manifest.json') -Encoding UTF8
& (Join-Path $runtime 'llama-server.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'Runtime did not start. Check CUDA driver compatibility.' }
