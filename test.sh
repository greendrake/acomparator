#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

usage() {
    echo "Usage: $0 [--executable]"
    echo ""
    echo "Run the acomparator test suite."
    echo ""
    echo "Options:"
    echo "  --executable    Build the executable first and run tests against it"
    echo "  --help          Show this help message"
    echo ""
    echo "By default, tests run against the Python source via 'python -m acomparator.cli'."
}

USE_EXECUTABLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --executable)
            USE_EXECUTABLE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Activate virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: Virtual environment not found. Run build.sh first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

if [[ "$USE_EXECUTABLE" == "true" ]]; then
    echo "=== Building executable ==="
    ./build.sh

    echo ""
    echo "=== Running tests against executable ==="
    export ACOMPARATOR_EXECUTABLE="$SCRIPT_DIR/dist/acomparator"
else
    echo "=== Running tests against source ==="
fi

python -m pytest test/test_comparison.py -v
