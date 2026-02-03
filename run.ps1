Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"
$ReqFile = Join-Path $RootDir "requirements.txt"
$HashFile = Join-Path $VenvDir ".requirements.sha256"

$Python = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $Python = "py"
}

if (-not $Python) {
    Write-Host "Error: Python is not installed. Please install Python 3.10+ and try again."
    exit 1
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at $VenvDir..."
    & $Python -m venv $VenvDir
}

Write-Host "Activating virtual environment..."
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
& $ActivateScript

$ReqHash = & $Python -c "import hashlib; from pathlib import Path; data = Path('requirements.txt').read_bytes(); print(hashlib.sha256(data).hexdigest())"

$NeedsInstall = $true
if (Test-Path $HashFile) {
    $ExistingHash = Get-Content $HashFile -Raw
    if ($ExistingHash.Trim() -eq $ReqHash.Trim()) {
        $NeedsInstall = $false
    }
}

if ($NeedsInstall) {
    Write-Host "Installing dependencies..."
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r $ReqFile --upgrade --upgrade-strategy only-if-needed
    New-Item -ItemType Directory -Force -Path $VenvDir | Out-Null
    $ReqHash | Out-File -FilePath $HashFile -Encoding ascii
} else {
    Write-Host "Dependencies are up to date."
}

Write-Host "Starting application..."
if ($args.Count -eq 0) {
    & $Python -m app ui
} else {
    & $Python -m app @args
}
