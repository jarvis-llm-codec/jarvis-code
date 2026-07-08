#!/usr/bin/env sh
set -eu

# Resolve symlinks so an installed `jarvis` shim (e.g. ~/.local/bin/jarvis ->
# .../jarvis.sh, as created by install.sh) still locates the real repo root.
SCRIPT_PATH=$0
while [ -h "$SCRIPT_PATH" ]; do
  script_dir=$(dirname "$SCRIPT_PATH")
  link=$(readlink "$SCRIPT_PATH")
  case $link in
    /*) SCRIPT_PATH=$link ;;
    *) SCRIPT_PATH=$script_dir/$link ;;
  esac
done
ROOT_DIR=$(CDPATH= cd "$(dirname "$SCRIPT_PATH")" && pwd -P)

find_python() {
  # Keep the candidate list in sync with install.sh: PATH order can hide a
  # new-enough interpreter behind an old one.
  for candidate in python3 python python3.13 python3.12 python3.11 python3.10 \
    /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON_BIN=$(find_python); then
  echo "JARVIS Code requires Python 3.10 or newer." >&2
  echo "Install Python 3.10+ and retry. On macOS, Homebrew users can run: brew install python" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/jarvis-launcher.py" "$@"
