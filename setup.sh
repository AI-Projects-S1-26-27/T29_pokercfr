#!/usr/bin/env bash
# One-command setup. Pins Python 3.12: torch and open_spiel have no 3.14 wheels.
set -euo pipefail
command -v uv >/dev/null || { echo "install uv first: brew install uv"; exit 1; }
uv venv --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -e ".[verify,deep,dev]"
echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then verify with:     pytest -q"
