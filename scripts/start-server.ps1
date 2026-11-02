param([string]$Binary = '', [string]$Model = '', [int]$Context = 16384, [int]$GpuLayers = 99)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Binary) { $Binary = Join-Path $repo 'tools\llama.cpp\llama-server.exe' }
if (-not $Model) { $Model = Join-Path $repo 'models\Qwen3.5-9B-Q4_K_M.gguf' }
$python = Join-Path $repo '.venv\Scripts\python.exe'
Push-Location -LiteralPath $repo
try {
    & $python -m datamate_bench preflight --require-cuda
    if ($LASTEXITCODE -ne 0) { throw 'CUDA unavailable. Connect power and enable NVIDIA GPU first.' }
    if (-not (Test-Path -LiteralPath $Binary)) { throw 'Missing llama-server.exe. Use official Windows CUDA 12.8+ build; pass -Binary or extract to tools/llama.cpp.' }
    if (-not (Test-Path -LiteralPath $Model)) { throw 'Run scripts/prepare-model.ps1 first.' }
    if ($Context -lt 4096) { throw 'Context must be at least 4096.' }
    $logs = Join-Path $repo 'artifacts'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $log = Join-Path $logs ('server_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
    [PSCustomObject]@{
        binary=[IO.Path]::GetFullPath($Binary); model=[IO.Path]::GetFullPath($Model);
        requested_context=$Context; requested_gpu_layers=$GpuLayers; parallel=1;
        batch_size=512; ubatch_size=128; reasoning_budget=0; log=$log;
        started_utc=[DateTime]::UtcNow.ToString('o');
        note='Requested settings; inspect log for actual allocation/offload.'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logs 'server_session.json') -Encoding UTF8
    Write-Host "Keep this terminal open. Log: $log"
    # Windows PowerShell 5 wraps native stderr logs in ErrorRecord objects.
    # Preserve those logs without treating a normal startup message as a fatal error.
    $savedErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Binary --model $Model --alias datamate-local --host 127.0.0.1 --port 8080 `
            --ctx-size $Context --n-gpu-layers $GpuLayers --parallel 1 `
            --batch-size 512 --ubatch-size 128 --jinja --reasoning-budget 0 2>&1 |
            ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
        $serverExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedErrorAction }
    if ($serverExitCode -ne 0) { throw 'Server error; check log. No CPU fallback is launched by this script.' }
} finally { Pop-Location }
