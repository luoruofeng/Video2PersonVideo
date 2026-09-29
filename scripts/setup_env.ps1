<#
.SYNOPSIS
    一键创建虚拟环境并安装 Video2PersonVideo 的全部依赖。

.EXAMPLE
    # CPU 环境（无独显）
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

.EXAMPLE
    # NVIDIA 显卡 + CUDA，并额外安装开发/打包依赖
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Backend cuda -Dev
#>
[CmdletBinding()]
param(
    [string] $VenvDir = ".venv",
    [ValidateSet("cpu", "cuda")] [string] $Backend = "cpu",
    # CUDA 版本对应的索引地址，请到 https://pytorch.org/get-started/locally/ 确认最新值
    [string] $TorchIndexUrl = "https://download.pytorch.org/whl/cu129",
    [switch] $Dev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==> 项目目录：$ProjectRoot" -ForegroundColor Cyan

# 1) 创建虚拟环境
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "==> 创建虚拟环境 $VenvDir" -ForegroundColor Cyan
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.13 -m venv $VenvDir
    }
    else {
        python -m venv $VenvDir
    }
}
else {
    Write-Host "==> 复用已存在的虚拟环境 $VenvDir" -ForegroundColor Cyan
}

$Python = Join-Path $VenvDir "Scripts\python.exe"

# 2) 升级 pip
& $Python -m pip install --upgrade pip

# 3) 安装 PyTorch（按 CPU / CUDA 选择不同 wheel 源）
if ($Backend -eq "cuda") {
    Write-Host "==> 安装 CUDA 版 PyTorch（$TorchIndexUrl）" -ForegroundColor Cyan
    & $Python -m pip install torch torchvision --index-url $TorchIndexUrl
}
else {
    Write-Host "==> 安装 CPU 版 PyTorch" -ForegroundColor Cyan
    & $Python -m pip install torch torchvision
}

# 4) 安装其余依赖
$Requirements = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
Write-Host "==> 安装依赖 $Requirements" -ForegroundColor Cyan
& $Python -m pip install -r $Requirements

# 5) 以可编辑模式安装本项目（提供 v2pv 命令）
Write-Host "==> 安装项目本体（可编辑模式）" -ForegroundColor Cyan
& $Python -m pip install -e .

# 6) 自检
Write-Host "==> 环境自检" -ForegroundColor Cyan
& $Python -m video2personvideo --check

Write-Host ""
Write-Host "完成。激活环境：$VenvDir\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "试跑命令    ：v2pv -i data\input\你的视频.mp4" -ForegroundColor Green
