@echo off
setlocal

cd /d "%~dp0"

echo === AudioSR Upscaling - Installation ===

if not exist "venv" (
    echo Error: Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Installing AudioSR sub-dependencies...
pip install -r requirements_audiosr.txt

echo Installing AudioSR (from git, with --no-deps to avoid version conflicts)...
pip install --no-deps git+https://github.com/haoheliu/versatile_audio_super_resolution.git

echo.
echo === AudioSR installation complete! ===
echo You can now use the AudioSR upscaling features in HeartMuse.
pause
