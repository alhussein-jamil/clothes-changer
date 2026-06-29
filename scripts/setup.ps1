#Requires -Version 5.1
<#
.SYNOPSIS
    Clothes Changer (clotheless-next) — Windows development setup.

.DESCRIPTION
    Creates a Python 3.10 virtual environment with uv, installs the package,
    CUDA PyTorch, ONNX Runtime GPU, and downloads default model weights.

.REQUIREMENTS
    - Python 3.10+ on PATH
    - Git
    - NVIDIA GPU drivers (optional; CPU fallback works for dev)

.USAGE
    cd D:\Projects\clothing_proj\clotheless-next
    powershell -ExecutionPolicy RemoteSigned -File scripts\setup.ps1

    Optional:
      -SkipModels   Skip model download (faster, offline)
      -SkipCuda     Skip CUDA torch / onnxruntime-gpu (CPU-only dev)
#>

param(
    [switch]$SkipModels,
    [switch]$SkipCuda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

function Write-Header($msg) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Yellow
}

function Assert-Command($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "[ERROR] '$cmd' not found on PATH." -ForegroundColor Red
        exit 1
    }
}

Write-Header 'Clothes Changer - Windows Setup'

Write-Step 1 6 "Checking prerequisites..."
Assert-Command "python"
$pyVersion = python --version 2>&1
Write-Host "    Python: $pyVersion"

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host '    uv not found - installing via pip...'
    python -m pip install uv -q
}
Assert-Command "uv"
Write-Host "    uv: $(uv --version)"

Write-Step 2 6 "Creating virtual environment (.venv)..."
if (-not (Test-Path ".venv")) {
    uv venv --python 3.10 .venv
} else {
    Write-Host "    .venv already exists, skipping creation."
}

$python = ".\.venv\Scripts\python.exe"
$uvPip = "uv pip install --python $python"

Write-Step 3 6 "Installing package (editable + dev extras)..."
Invoke-Expression "$uvPip -e `".[dev]`""
if ($LASTEXITCODE -ne 0) { exit 1 }

if (-not $SkipCuda) {
    Write-Step 4 6 "Installing CUDA PyTorch + ONNX Runtime GPU..."
    & (Join-Path $ProjectRoot "scripts\install-cuda.ps1")
    if ($LASTEXITCODE -ne 0) { exit 1 }
    if (-not (Select-String -Path .env -Pattern "XFORMERS_FORCE_DISABLE_TRITON" -Quiet)) {
        Add-Content .env "`nXFORMERS_FORCE_DISABLE_TRITON=1"
    }
    Invoke-Expression "$uvPip gdown"
} else {
    Write-Step 4 6 'Skipping CUDA packages (-SkipCuda).'
}

Write-Step 5 6 "Creating local config..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "    Created .env from .env.example"
} else {
    Write-Host "    .env already exists."
}

if (-not $SkipModels) {
    Write-Step 6 6 "Downloading default models..."
    & $python -m clothes_changer.scripts.download_models
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[WARN] Model download failed - run manually: .\.venv\Scripts\python.exe -m clothes_changer.scripts.download_models' -ForegroundColor Yellow
    }
} else {
    Write-Step 6 6 'Skipping model download (-SkipModels).'
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Run app:   .\.venv\Scripts\python.exe -m clothes_changer.main"
Write-Host "  Debug:     set CLOTHES_CHANGER_PIPELINE_DEBUG=true in .env"
Write-Host ""
