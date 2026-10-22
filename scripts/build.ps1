param(
    [ValidateSet('cuda', 'cpu')][string]$Variant = 'cuda',
    [switch]$SkipInstall,
    [switch]$OneFile
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize the pinned MeanVC2 submodule; use a Git checkout' }
    if (-not (Test-Path -LiteralPath 'third_party/MeanVC2/runtime/run_rt.py')) { throw 'MeanVC2 runtime is missing' }
    $venvName = if ($Variant -eq 'cuda') { '.venv' } else { '.venv-cpu' }
    $pythonExe = Join-Path $projectRoot "$venvName/Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        py -3.11 -m venv $venvName
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 x64 is required' }
    }
    if (-not $SkipInstall) {
        & $pythonExe -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw 'pip installation failed' }
        $wheelIndex = if ($Variant -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu121' }
        & $pythonExe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url $wheelIndex
        if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
        & $pythonExe -m pip install -c requirements-lock.txt -e '.[build]'
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    }
    & $pythonExe -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    $buildName = if ($OneFile) { "$Variant-onefile" } else { $Variant }
    $specArguments = if ($OneFile) { @('--', '--onefile') } else { @() }
    & $pythonExe -m PyInstaller --noconfirm --clean --workpath "build/$buildName" --distpath "dist/$buildName" MeanVC2Desktop.spec @specArguments
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
    $bundle = Join-Path $projectRoot "dist/$buildName"
    if (-not $OneFile) {
        $bundle = Join-Path $bundle 'MeanVC2Desktop'
        Copy-Item -LiteralPath README.md -Destination $bundle
        Copy-Item -LiteralPath THIRD_PARTY.md -Destination $bundle
    }
    if (-not $OneFile -and (Test-Path -LiteralPath 'reports/VALIDATION.md')) {
        $bundleReports = Join-Path $bundle 'reports'
        New-Item -ItemType Directory -Force -Path $bundleReports | Out-Null
        Get-ChildItem -LiteralPath reports -File | Where-Object { $_.Extension -in @('.md', '.json', '.png') } | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $bundleReports
        }
    }
    $exePath = Join-Path $bundle 'MeanVC2Desktop.exe'
    $diagnostics = Join-Path $bundle 'build-diagnostics.json'
    $process = Start-Process -FilePath $exePath -ArgumentList @('--diagnostics', ('"' + $diagnostics + '"')) -PassThru -WindowStyle Hidden
    if (-not $process.WaitForExit(300000)) {
        Stop-Process -Id $process.Id
        throw 'Frozen executable diagnostics timed out'
    }
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $diagnostics)) { throw 'Frozen executable diagnostics failed' }
    Write-Output "Built: $exePath"
} finally {
    Pop-Location
}
