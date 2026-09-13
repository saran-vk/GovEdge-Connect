#!/usr/bin/env bash
# setup_venv.sh — Create Python virtual environment and install dependencies.
# Linux/macOS equivalent of scripts/setup_venv.ps1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== Setting up virtual environment in $VENV_DIR ==="

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv."
else
    echo "Venv already exists — reusing."
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

echo ""
echo "=== Done. Activate with:  source .venv/bin/activate ==="
