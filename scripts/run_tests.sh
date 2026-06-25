#!/usr/bin/env bash
# run_tests.sh — Run the EXoPLORE test suite in the project venv.
#
# Usage (from the repo root):
#   cd /path/to/EXoPLORE_github
#   bash scripts/run_tests.sh
#
# Or equivalently, from within the activated venv:
#   source .venv/bin/activate
#   export SDKROOT="$(xcrun --show-sdk-path)"
#   python -m pytest

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        echo "Activating venv..."
        source .venv/bin/activate
    else
        echo "WARNING: No .venv found. Running with system Python."
    fi
fi

# macOS: set SDKROOT to avoid Fortran/linker issues
if [[ "$(uname)" == "Darwin" ]]; then
    export SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || echo '')"
fi

echo "Python: $(python --version)"
echo "pytest: $(python -m pytest --version)"
echo ""

python -m pytest tests/ -v "$@"
