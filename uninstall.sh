#!/usr/bin/env sh
set -eu

INSTALL_DIR=${JARVIS_CODE_INSTALL_DIR:-"$HOME/.local/share/jarvis-code"}
BIN_DIR=${JARVIS_CODE_BIN_DIR:-"$HOME/.local/bin"}
USER_DATA_DIR=${JARVIS_CODE_USER_DATA_DIR:-"$HOME/.jarvis-code"}
REMOVE_USER_DATA=${JARVIS_CODE_REMOVE_USER_DATA:-0}
REMOVE_MODEL_CACHE=${JARVIS_CODE_REMOVE_MODEL_CACHE:-0}
KEEP_COMMAND=${JARVIS_CODE_KEEP_COMMAND:-0}

log() {
  printf '[jarvis-uninstall] %s\n' "$1"
}

usage() {
  cat <<'EOF'
Usage: uninstall.sh [options]

Options:
  --install-dir DIR       JARVIS Code install directory
  --bin-dir DIR           directory containing the jarvis command shim
  --remove-user-data      also remove ~/.jarvis-code
  --remove-model-cache    also remove the local BAAI/bge-m3 Hugging Face cache
  --keep-command          keep the jarvis command shim
  -h, --help              show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR=$2
      shift 2
      ;;
    --bin-dir)
      BIN_DIR=$2
      shift 2
      ;;
    --remove-user-data)
      REMOVE_USER_DATA=1
      shift
      ;;
    --remove-model-cache)
      REMOVE_MODEL_CACHE=1
      shift
      ;;
    --keep-command)
      KEEP_COMMAND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

normalize_path() {
  path=$1
  if [ -z "$path" ]; then
    printf '\n'
    return
  fi
  if [ -d "$path" ]; then
    (cd "$path" && pwd -P)
  else
    parent=$(dirname "$path")
    name=$(basename "$path")
    if [ -d "$parent" ]; then
      printf '%s/%s\n' "$(cd "$parent" && pwd -P)" "$name"
    else
      printf '%s\n' "$path"
    fi
  fi
}

assert_safe_delete_path() {
  path=$(normalize_path "$1")
  purpose=$2
  if [ -z "$path" ]; then
    printf 'Refusing to delete empty %s path.\n' "$purpose" >&2
    exit 1
  fi
  case "$path" in
    /|"$HOME"|"$HOME/.local"|"$HOME/.local/share"|"$HOME/.cache"|"$HOME/.cache/huggingface")
      printf 'Refusing to delete protected %s path: %s\n' "$purpose" "$path" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "$path"
}

is_jarvis_install_dir() {
  dir=$1
  if [ ! -e "$dir" ]; then
    return 1
  fi
  if [ -f "$dir/jarvis.ps1" ] && [ -d "$dir/sidecar" ] && [ -d "$dir/pi" ]; then
    return 0
  fi
  leaf=$(basename "$(normalize_path "$dir")")
  if { [ "$leaf" = "JARVIS-Code" ] || [ "$leaf" = "jarvis-code" ]; } &&
    { [ -f "$dir/install.ps1" ] || [ -f "$dir/install.sh" ] || [ -f "$dir/jarvis.sh" ]; }; then
    return 0
  fi
  return 1
}

is_jarvis_user_data_dir() {
  dir=$1
  if [ ! -e "$dir" ]; then
    return 1
  fi
  leaf=$(basename "$(normalize_path "$dir")")
  if [ "$leaf" = ".jarvis-code" ]; then
    return 0
  fi
  for name in config.yaml providers.yaml auth.json workspaceMemory conversation raw-store; do
    if [ -e "$dir/$name" ]; then
      return 0
    fi
  done
  return 1
}

stop_sidecar() {
  port=${JARVIS_SIDECAR_PORT:-8765}
  pids=
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  elif command -v pgrep >/dev/null 2>&1; then
    pids=$(pgrep -f jarvis_sidecar 2>/dev/null || true)
  fi
  if [ -z "$pids" ]; then
    return
  fi
  for pid in $pids; do
    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    case "$cmd" in
      *jarvis_sidecar*)
        log "stopping existing JARVIS sidecar process $pid"
        kill "$pid" 2>/dev/null || true
        ;;
    esac
  done
}

remove_command_shim() {
  if [ "$KEEP_COMMAND" = "1" ]; then
    log "keeping command shim"
    return
  fi
  shim="$BIN_DIR/jarvis"
  if [ ! -e "$shim" ] && [ ! -L "$shim" ]; then
    return
  fi
  if [ -L "$shim" ]; then
    target=$(readlink "$shim" 2>/dev/null || true)
    case "$target" in
      "$INSTALL_DIR"/jarvis.sh|"$INSTALL_DIR"/*)
        rm -f "$shim"
        log "removed command shim $shim"
        ;;
      *)
        log "kept $shim because it does not point at $INSTALL_DIR"
        ;;
    esac
  else
    log "kept non-symlink command at $shim"
  fi
}

bge_m3_cache_dir() {
  if [ -n "${HF_HUB_CACHE:-}" ]; then
    printf '%s/models--BAAI--bge-m3\n' "$HF_HUB_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then
    printf '%s/hub/models--BAAI--bge-m3\n' "$HF_HOME"
  else
    printf '%s/.cache/huggingface/hub/models--BAAI--bge-m3\n' "$HOME"
  fi
}

stop_sidecar
install_path=$(assert_safe_delete_path "$INSTALL_DIR" "JARVIS Code install")

if [ -e "$install_path" ]; then
  if ! is_jarvis_install_dir "$install_path"; then
    printf "Refusing to delete '%s' because it does not look like a JARVIS Code install directory.\n" "$install_path" >&2
    exit 1
  fi
fi

remove_command_shim

if [ -e "$install_path" ]; then
  rm -rf "$install_path"
  log "removed install directory $install_path"
else
  log "install directory not found: $install_path"
fi

if [ "$REMOVE_USER_DATA" = "1" ]; then
  user_data_path=$(assert_safe_delete_path "$USER_DATA_DIR" "JARVIS Code user data")
  if [ -e "$user_data_path" ]; then
    if ! is_jarvis_user_data_dir "$user_data_path"; then
      printf "Refusing to delete '%s' because it does not look like JARVIS Code user data.\n" "$user_data_path" >&2
      exit 1
    fi
    rm -rf "$user_data_path"
    log "removed user data $user_data_path"
  else
    log "user data directory not found: $user_data_path"
  fi
else
  log "kept user data at $USER_DATA_DIR"
fi

if [ "$REMOVE_MODEL_CACHE" = "1" ]; then
  model_cache=$(assert_safe_delete_path "$(bge_m3_cache_dir)" "bge-m3 model cache")
  if [ "$(basename "$model_cache")" != "models--BAAI--bge-m3" ]; then
    printf 'Refusing to delete unexpected model cache path: %s\n' "$model_cache" >&2
    exit 1
  fi
  if [ -e "$model_cache" ]; then
    rm -rf "$model_cache"
    log "removed bge-m3 model cache $model_cache"
  else
    log "bge-m3 model cache not found: $model_cache"
  fi
else
  log "kept Hugging Face model cache"
fi

log "uninstalled JARVIS Code"
log "system prerequisites such as Node.js, Python, Git, and VC++ Redistributable were not removed"
