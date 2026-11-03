param([string]$TorchPython = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $TorchPython) {
    $TorchPython = Join-Path (Split-Path -Parent $repo) 'venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $TorchPython)) {
    throw 'Pass -TorchPython pointing to an existing Python with a working CUDA PyTorch build.'
}
Push-Location -LiteralPath $repo
try {
    if (-not (Test-Path -LiteralPath '.train-venv\Scripts\python.exe')) {
        & $TorchPython -m venv .train-venv
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create training venv.' }
    }
    $runtimeSite = (& $TorchPython -c "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('torch').origin).resolve().parent.parent)").Trim()
    if (-not (Test-Path -LiteralPath (Join-Path $runtimeSite 'torch'))) { throw 'Selected Python has no torch package.' }
    [IO.File]::WriteAllText((Join-Path $repo '.train-venv\Lib\site-packages\local_torch_runtime.pth'), $runtimeSite + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    & '.\.train-venv\Scripts\python.exe' -m pip install --disable-pip-version-check -r requirements-training.txt
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
    & '.\.train-venv\Scripts\python.exe' -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA is unavailable in the training environment.' }
} finally { Pop-Location }
