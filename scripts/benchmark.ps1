param([int]$Limit = 10, [int]$MaxSteps = 12, [int]$QuestionTimeout = 240)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
try {
    $python = '.\.venv\Scripts\python.exe'
    & $python -m datamate_bench preflight --require-cuda
    if ($LASTEXITCODE -ne 0) { throw 'GPU unavailable. Benchmark not started.' }
    & $python -m datamate_bench speed --model datamate-local
    if ($LASTEXITCODE -ne 0) { throw 'Speed probe failed. Check the server terminal.' }
    & $python -m datamate_bench run --model datamate-local --require-cuda --limit $Limit --max-steps $MaxSteps --question-timeout $QuestionTimeout
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed. See artifacts for partial outputs/logs.' }
} finally { Pop-Location }
