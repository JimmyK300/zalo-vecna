param(
    [string]$ModelRepo = 'unsloth/Qwen3.5-9B-GGUF',
    [string]$FileName = 'Qwen3.5-9B-Q4_K_M.gguf',
    [string]$Revision = '3885219b6810b007914f3a7950a8d1b469d598a5'
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ([IO.Path]::GetFileName($FileName) -ne $FileName -or -not $FileName.EndsWith('.gguf')) { throw 'FileName must be a GGUF basename.' }
$directory = Join-Path $repo 'models'
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$revisionEscaped = [uri]::EscapeDataString($Revision)
$metadata = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$ModelRepo/revision/$revisionEscaped`?blobs=true"
$file = @($metadata.siblings | Where-Object { $_.rfilename -eq $FileName })
if ($file.Count -ne 1 -or -not $file[0].lfs.sha256) { throw 'Missing unique file with LFS SHA-256.' }
$expected = $file[0].lfs.sha256
$destination = Join-Path $directory $FileName
if (Test-Path -LiteralPath $destination) {
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Existing file has a different hash. Move it aside before downloading.' }
} else {
    $partial = $destination + '.part'
    $url = "https://huggingface.co/$ModelRepo/resolve/$($metadata.sha)/$FileName"
    Write-Host "Downloading model (about 5.68 GB for default). No Colab CU is used."
    & curl.exe --location --fail --silent --show-error --connect-timeout 20 --max-time 1800 --retry 3 --retry-all-errors --speed-time 45 --speed-limit 1024 --continue-at - --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw 'Download failed; partial retained for resume.' }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Checksum mismatch.' }
    Move-Item -LiteralPath $partial -Destination $destination
}
[PSCustomObject]@{model_repo=$ModelRepo; revision=$metadata.sha; filename=$FileName; sha256=$expected;
    note='Quantized local baseline. Verify base-model provenance before a competition submission.'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $directory 'model_manifest.json') -Encoding UTF8
Write-Host "Verified model: $destination"
