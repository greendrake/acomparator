#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
DIST_DIR="dist"

echo "=== acomparator Build Script ==="

# Check for Python 3.11+
PYTHON=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
        major=$("$cmd" -c 'import sys; print(sys.version_info.major)')
        if [[ "$major" -eq 3 && "$version" -ge 11 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "Error: Python 3.11+ is required"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# Create virtual environment if needed
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e . -q
pip install pyinstaller -q

# Clean previous build
echo "Cleaning previous build..."
rm -rf build dist *.spec

# Build executable
# - --log-level ERROR hides PyInstaller warnings about optional dependencies
echo "Building executable..."
PYTHONWARNINGS=ignore pyinstaller \
    --onefile \
    --name acomparator \
    --hidden-import=acomparator \
    --collect-all librosa \
    --collect-all audioread \
    --log-level ERROR \
    acomparator/cli.py

# Clean up intermediate build artifacts
rm -rf build *.spec

# Show result
echo ""
echo "=== Build Complete ==="
ls -lh "$DIST_DIR/acomparator"
echo ""
echo "Executable: $DIST_DIR/acomparator"
echo ""
echo "To test: ./dist/acomparator --help"
