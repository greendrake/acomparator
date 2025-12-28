#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

usage() {
    echo "Usage: $0 [--check] [--fix]"
    echo ""
    echo "Run code formatting and linting with ruff."
    echo ""
    echo "Options:"
    echo "  --check    Check for issues without fixing (default)"
    echo "  --fix      Automatically fix issues and format code"
    echo "  --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Check for issues"
    echo "  $0 --fix        # Fix issues and format code"
}

MODE="check"

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            MODE="check"
            shift
            ;;
        --fix)
            MODE="fix"
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

# Ensure ruff is installed
if ! command -v ruff &> /dev/null; then
    echo "Installing ruff..."
    pip install -e ".[dev]" -q
fi

if [[ "$MODE" == "fix" ]]; then
    echo "=== Formatting code ==="
    ruff format acomparator test

    echo ""
    echo "=== Fixing lint issues ==="
    ruff check --fix acomparator test

    echo ""
    echo "=== Done ==="
else
    echo "=== Checking formatting ==="
    ruff format --check acomparator test

    echo ""
    echo "=== Checking lint issues ==="
    ruff check acomparator test

    echo ""
    echo "=== All checks passed ==="
fi
