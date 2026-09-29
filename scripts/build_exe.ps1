<#
.SYNOPSIS
    用 PyInstaller 把项目打包成 Windows 独立 exe。

.EXAMPLE
    # 目录版（启动快，推荐）
    powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1

.EXAMPLE
    # 单文件版（体积大、启动慢，但只有一个 exe 便于分发）
    powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 -OneFile
#>
[CmdletBinding()]
param(
    [string] $VenvDir = ".venv",
    [switch] $OneFile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "未找到虚拟环境 $VenvDir，请先执行 scripts\setup_env.ps1 -Dev"
}

$env:V2PV_ONEFILE = if ($OneFile) { "1" } else { "0" }

Write-Host "==> 开始打包（OneFile=$OneFile）" -ForegroundColor Cyan
& $Python -m PyInstaller --noconfirm --clean "build\Video2PersonVideo.spec"

Write-Host ""
Write-Host "打包完成，产物在 dist\ 目录下。" -ForegroundColor Green
Write-Host "首次运行会自动从网上下载 YOLO 权重（yolo11n.pt，约 5MB），请保持联网。" -ForegroundColor Yellow
