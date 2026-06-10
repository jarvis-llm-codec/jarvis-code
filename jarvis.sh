#!/usr/bin/env sh
set -eu

# Resolve symlinks so an installed `jarvis` shim (e.g. ~/.local/bin/jarvis ->
# .../jarvis.sh, as created by install.sh) still locates the real repo root.
SCRIPT_PATH=$0
while [ -h "$SCRIPT_PATH" ]; do
  link=$(readlink "$SCRIPT_PATH")
  case $link in
    /*) SCRIPT_PATH=$link ;;
    *) SCRIPT_PATH=$(dirname -- "$SCRIPT_PATH")/$link ;;
  esac
done
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "JARVIS Code requires Python 3.10 or newer." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/jarvis-launcher.py" "$@"
