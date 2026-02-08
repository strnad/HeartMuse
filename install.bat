@echo off
setlocal

cd /d "%~dp0"

echo === HeartMuse Music Generator - Installation ===

where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

python --version

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA 12.4 support...
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

if not exist "heartlib" (
    echo Cloning heartlib...
    git clone https://github.com/HeartMuLa/heartlib.git
)

echo Installing heartlib...
pip install -e heartlib

echo Installing app dependencies...
pip install -r requirements.txt

echo.
echo === Installation complete! ===
echo Run run.bat to start the application.
echo.
echo Optional: Run install_audiosr.bat to enable AudioSR upscaling (48kHz super-resolution).
pause
