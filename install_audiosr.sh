#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== AudioSR Upscaling - Installation ==="

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi

source venv/bin/activate

echo "Installing AudioSR sub-dependencies..."
pip install -r requirements_audiosr.txt

echo "Installing AudioSR (from git, with --no-deps to avoid version conflicts)..."
pip install --no-deps git+https://github.com/haoheliu/versatile_audio_super_resolution.git

echo ""
echo "=== AudioSR installation complete! ==="
echo "You can now use the AudioSR upscaling features in HeartMuse."
