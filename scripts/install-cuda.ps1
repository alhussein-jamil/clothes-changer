#Requires -Version 5.1
<#
.SYNOPSIS
    Reinstall CUDA PyTorch + ONNX Runtime GPU (Windows).

.DESCRIPTION
    PyPI / uv sync installs CPU-only torch. Run this after setup, uv sync, or
    any dependency reinstall that touches torch.

.USAGE
    powershell -ExecutionPolicy RemoteSigned -File scripts\install-cuda.ps1
#>

param(
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu130"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "[ERROR] .venv not found. Run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$uvPip = "uv pip install --python $python"

Write-Host "Installing CUDA PyTorch from $CudaIndex ..." -ForegroundColor Cyan
Invoke-Expression "$uvPip torch torchvision --index-url $CudaIndex --reinstall"
Invoke-Expression "$uvPip xformers --index-url $CudaIndex --no-deps"

# Gradio 4.42 requires pillow<11; CUDA torch may pull a newer pillow.
Invoke-Expression "$uvPip `"pillow>=10.0.0,<11.0`""

Write-Host "Installing ONNX Runtime GPU ..." -ForegroundColor Cyan
Invoke-Expression "$uvPip uninstall onnxruntime -y"
Invoke-Expression "$uvPip onnxruntime-gpu nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12"

& $python -c @"
import torch
print(f'torch {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu: {torch.cuda.get_device_name(0)}')
"@
