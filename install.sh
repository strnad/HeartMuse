#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== HeartMuse Music Generator - Installation ==="

# Check Python
PYTHON=""
for cmd in python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python not found. Please install Python 3.10+"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# Clone heartlib
if [ ! -d "heartlib" ]; then
    echo "Cloning heartlib..."
    git clone https://github.com/HeartMuLa/heartlib.git
fi

echo "Installing heartlib..."
pip install -e ./heartlib

echo "Installing app dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Installation complete! ==="
echo "Run ./run.sh to start the application."
echo ""
echo "Optional: Run ./install_audiosr.sh to enable AudioSR upscaling (48kHz super-resolution)."
