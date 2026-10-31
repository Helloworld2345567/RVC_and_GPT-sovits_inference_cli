[CmdletBinding()]
param(
    [ValidateSet('CU129', 'CPU')]
    [string]$Device = 'CU129'
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'FFmpeg must be available on PATH.' }
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) { throw 'ffprobe must be available on PATH.' }

$env:UV_CACHE_DIR = Join-Path $projectRoot '.codex/uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.codex/python'
$env:NLTK_DATA = Join-Path $projectRoot 'GPT-SoVITS/.codex/runtime-cache/nltk_data'
$venvPath = Join-Path $projectRoot '.codex/venv'
$pythonPath = Join-Path $venvPath 'Scripts/python.exe'
$env:TEMP = Join-Path $projectRoot '.codex/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}
function Invoke-ProjectPython {
    & $pythonPath @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Invoke-Uv venv --python 3.12 --seed $venvPath
}
Invoke-ProjectPython -c "import sys; assert sys.version_info[:2] == (3, 12), 'The shared environment requires Python 3.12.'"

# The Japanese extension is built for NumPy 2 instead of using an old ABI wheel.
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $vsPath = & $vswhere -latest -prerelease -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            Import-Module (Join-Path $vsPath 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
            Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
            $env:DISTUTILS_USE_SDK = '1'
        }
    }
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    & $pythonPath -c 'import pyopenjtalk' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Install Visual Studio C++ Build Tools with the Windows SDK to build PyOpenJTalk for NumPy 2.' }
}

$backend = if ($Device -eq 'CU129') { 'cu129' } else { 'cpu' }
Invoke-Uv pip install --python $pythonPath "torch==2.8.0+$backend" "torchaudio==2.8.0+$backend" --index-url "https://download.pytorch.org/whl/$backend"
Invoke-Uv pip install --python $pythonPath -r (Join-Path $projectRoot 'requirements-lock.txt') --index-url https://pypi.org/simple
& $pythonPath -c 'import pyopenjtalk' 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Uv pip install --python $pythonPath --reinstall-package pyopenjtalk --no-binary pyopenjtalk -r (Join-Path $projectRoot 'requirements-lock.txt') --index-url https://pypi.org/simple
}
Invoke-Uv pip install --python $pythonPath --no-deps -e (Join-Path $projectRoot 'RVC')
Invoke-ProjectPython -m pip check

New-Item -ItemType Directory -Force $env:NLTK_DATA | Out-Null
Invoke-ProjectPython (Join-Path $projectRoot 'scripts/prepare_language_resources.py')
if ($Device -eq 'CU129') {
    Invoke-ProjectPython -c "import torch; assert torch.version.cuda == '12.9', torch.version.cuda; assert torch.cuda.is_available(), 'CUDA is not available'; a = torch.ones(2, 2, device='cuda'); assert (a @ a).sum().item() == 8; print(torch.cuda.get_device_name())"
}
Write-Host "Shared environment ready. Activate with: . '$venvPath/Scripts/Activate.ps1'"
