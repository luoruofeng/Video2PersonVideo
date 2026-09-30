<#
.SYNOPSIS
    一键构建 Windows 安装包 Video2PersonVideo-Setup.exe。

.DESCRIPTION
    分两步：
      ① 先把项目打成 wheel（放进 build\installer_payload\），
        安装器装的时候直接装这个 wheel，不需要目标机器有 setuptools；
      ② 再用 PyInstaller 把"安装器"打包成单文件 exe。

    产出的 Setup.exe 只有几十 MB：PyTorch / YOLO / OpenCV 都不在里面，
    它们由用户在图形安装向导里按自己的显卡现下（带断点续传）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1

.EXAMPLE
    # 目录版（启动更快、便于排错）+ 控制台输出（调试用）
    powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1 -Dir -Console
#>
[CmdletBinding()]
param(
    [string] $VenvDir = ".venv",
    # 打包成目录版（默认单文件，便于分发）
    [switch] $Dir,
    # 保留控制台窗口（默认无控制台，纯图形界面）
    [switch] $Console,
    # 跳过 wheel 构建（复用上一次的 build\installer_payload）
    [switch] $SkipWheel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "未找到虚拟环境 $VenvDir，请先执行：scripts\setup_env.ps1 -Dev"
}

$PayloadDir = Join-Path $ProjectRoot "build\installer_payload"

if (-not $SkipWheel) {
    Write-Host "==> 构建项目 wheel（安装器用它安装程序本体）" -ForegroundColor Cyan
    if (Test-Path $PayloadDir) { Remove-Item $PayloadDir -Recurse -Force }
    New-Item -ItemType Directory -Path $PayloadDir -Force | Out-Null
    & $Python -m pip wheel . --no-deps --no-build-isolation --wheel-dir $PayloadDir
    if ($LASTEXITCODE -ne 0) { throw "wheel 构建失败" }
    Get-ChildItem $PayloadDir -Filter *.whl | ForEach-Object {
        Write-Host "    $($_.Name)（$([math]::Round($_.Length / 1MB, 2)) MB）" -ForegroundColor DarkGray
    }
}
else {
    Write-Host "==> 跳过 wheel 构建，复用 $PayloadDir" -ForegroundColor Yellow
}

$env:V2PV_SETUP_ONEFILE = if ($Dir) { "0" } else { "1" }
$env:V2PV_INSTALLER_CONSOLE = if ($Console) { "1" } else { "0" }

Write-Host "==> 打包安装器（单文件=$(-not $Dir)，控制台=$Console）" -ForegroundColor Cyan
& $Python -m PyInstaller --noconfirm --clean "build\Video2PersonVideo-Setup.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败" }

Write-Host ""
Write-Host "打包完成。产物：" -ForegroundColor Green
if ($Dir) {
    Write-Host "  dist\Video2PersonVideo-Setup\（把整个文件夹一起分发，运行里面的 exe）" -ForegroundColor Green
}
else {
    Write-Host "  dist\Video2PersonVideo-Setup.exe（单文件，直接发给用户）" -ForegroundColor Green
}
Write-Host ""
Write-Host "用户拿到后：双击 → 欢迎页 → 选目录 → 选组件（默认已按显卡推荐）→ 开始安装。" -ForegroundColor Yellow
Write-Host "安装过程中会断点续传下载 PyTorch / YOLO；中途关窗口也不会白下。" -ForegroundColor Yellow
