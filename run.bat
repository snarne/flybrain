@echo off
REM Start Flybrain on Windows. Everything it installs stays inside this folder
REM (hidden .tools\, .cache\ and .venv-windows-x64\), so the same folder also works on a Mac.
cd /d "%~dp0"
set "ROOT=%CD%"
set "P=windows-x64"
set "UV_CACHE_DIR=%ROOT%\.cache\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT%\.cache\python-%P%"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_PROJECT_ENVIRONMENT=%ROOT%\.venv-%P%"
set "HF_HOME=%ROOT%\.cache\huggingface"
set "TORCH_HOME=%ROOT%\.cache\torch"
set "NUMBA_CACHE_DIR=%ROOT%\.cache\numba-%P%"
set "PYTHONPYCACHEPREFIX=%ROOT%\.cache\pycache-%P%"
set "UV=%ROOT%\.tools\%P%\uv.exe"
if not exist "%UV%" (
  echo Setting up uv ^(a small Python manager^) in .tools\%P% ...
  powershell -NoProfile -ExecutionPolicy ByPass -Command "$ErrorActionPreference='Stop'; $d='%ROOT%\.tools\%P%'; New-Item -ItemType Directory -Force $d | Out-Null; $z=Join-Path $d 'uv.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $z; Expand-Archive -Force $z $d; Remove-Item $z; Get-ChildItem $d -Recurse -Filter uv.exe | Select-Object -First 1 | Move-Item -Destination $d -Force -ErrorAction SilentlyContinue"
)
if not exist "%UV%" (
  echo Couldn't download uv. Check the internet connection and run this again.
  pause
  exit /b 1
)
echo Starting Flybrain (%P%). The first run installs Python and packages; this takes a few minutes.
"%UV%" run --frozen --python 3.12 server\app.py
pause
