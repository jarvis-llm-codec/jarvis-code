#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "JARVIS Code requires Python 3.10 or newer." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/jarvis-launcher.py" "$@"
