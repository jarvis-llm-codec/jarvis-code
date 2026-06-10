from __future__ import annotations

import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROLE_CONFIG: dict[str, Any] = {
    "roles": {
        "chat": "ollama-cloud/glm-5-ollama",
        "subagent": "ollama-cloud/glm-5-ollama",
        "router": "ollama-cloud/glm-5-ollama",
        "encoder": "ollama-cloud/devstral-small-2-24b-cloud",
    },
    "encoder": {
        "providers": [
            {
                "name": "dashscope-coding-glm-5",
                "model": "glm-5",
                "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
                "api_key_env": "DASHSCOPE_CODING_API_KEY",
                "max_tokens": 12288,
            },
            {
                "name": "dashscope-coding-qwen3-coder-next",
                "model": "qwen3-coder-next",
                "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
                "api_key_env": "DASHSCOPE_CODING_API_KEY",
                "max_tokens": 12288,
            },
            {
                "name": "dashscope-coding-qwen3-max",
                "model": "qwen3-max-2026-01-23",
                "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
                "api_key_env": "DASHSCOPE_CODING_API_KEY",
                "max_tokens": 12288,
            },
            {
                "name": "ollama-glm-5.1-fallback",
                "model": "glm-5.1:cloud",
                "base_url": "http://localhost:11434/v1",
                "api_key_env": "",
                "max_tokens": 12288,
            },
        ]
    },
    "conversation_tail": {"enabled": False},
}

DEFAULT_PROVIDER_CONFIG: dict[str, Any] = {
    "defaults": {
        "primary": "devstral-small-2-24b-cloud",
        "fallback": [],
    },
    "providers": {
        "openai-codex": {
            "oauth_provider": "chatgpt",
            "oauth_token_path": "~/.jarvis-code/auth.json",
            "api_base": "https://chatgpt.com/backend-api/codex",
            "models": {
                "openai-codex-gpt-5.5": {
                    "litellm_id": "openai/gpt-5.5",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
                "openai-codex-gpt-5.4": {
                    "litellm_id": "openai/gpt-5.4",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
                "openai-codex-gpt-5.4-mini": {
                    "litellm_id": "openai/gpt-5.4-mini",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "cheap",
                },
                "openai-codex-gpt-5.3-codex": {
                    "litellm_id": "openai/gpt-5.3-codex",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
                "openai-codex-gpt-5.2-codex": {
                    "litellm_id": "openai/gpt-5.2-codex",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
                "openai-codex-gpt-5.1": {
                    "litellm_id": "openai/gpt-5.1",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
            },
        },
        "chatgpt-oauth": {
            "oauth_provider": "chatgpt",
            "oauth_token_path": "~/.jarvis-code/auth.json",
            "api_base": "https://chatgpt.com/backend-api/codex",
            "models": {
                "gpt-5": {
                    "litellm_id": "openai/gpt-5",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
            },
        },
        "anthropic": {
            "api_keys": ["${ANTHROPIC_API_KEY}", "${ANTHROPIC_API_KEY_2}"],
            "models": {
                "claude-sonnet-4-6": {
                    "litellm_id": "anthropic/claude-sonnet-4-6",
                    "cost_in_per_1m": 3.0,
                    "cost_out_per_1m": 15.0,
                    "tier": "quality",
                },
            },
        },
        "ollama-cloud": {
            "api_keys": ["${OLLAMA_API_KEY}"],
            "api_base": "https://ollama.com/v1",
            "models": {
                "glm-5-ollama": {
                    "litellm_id": "openai/glm-5",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "quality",
                },
                "devstral-small-2-24b-cloud": {
                    "litellm_id": "openai/devstral-small-2:24b",
                    "cost_in_per_1m": 0.0,
                    "cost_out_per_1m": 0.0,
                    "tier": "cheap",
                },
            },
        }
    }
}


def _repo_root() -> Path:
    """Repository root for this checkout. Sidecar lives at
    `<repo>/sidecar/jarvis_sidecar/`, so parents[2] is the repo top."""
    return Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    """User-home data directory. Keeps JARVIS memory stable across checkouts."""
    return Path("~/.jarvis-code").expanduser()


LEGACY_HOME_DIR = Path("~/.jarvis-code").expanduser()
SENTINEL_FILENAME = ".migrated"

# (legacy basename, new basename, kind). Order is meaningful: directories
# first so config files end up alongside them in the new layout.
LEGACY_MOVE_SPEC: tuple[tuple[str, str, str], ...] = (
    ("workspaceMemory", "workspaceMemory", "dir"),
    ("pi-sidecar", "raw-store", "dir"),
    ("config.yaml", "config.yaml", "file"),
    ("providers.yaml", "providers.yaml", "file"),
)


def perform_legacy_moves(
    legacy_home: Path,
    data_root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Copy each known legacy item from `legacy_home` into `data_root`.

    Skips items whose source is absent. Skips items whose destination already
    exists unless `force` is set (in which case the existing destination is
    moved aside to a timestamped `.bak.pre_phase2_*` sibling first). Returns
    a list of move records describing what happened (status one of:
    "copied", "skipped_no_source", "skipped_dest_exists", "would_copy",
    "forced_backup_then_copied").
    """
    records: list[dict[str, Any]] = []
    if not dry_run:
        data_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    for legacy_name, new_name, kind in LEGACY_MOVE_SPEC:
        src = legacy_home / legacy_name
        dst = data_root / new_name
        if not src.exists():
            records.append({"src": str(src), "dst": str(dst), "status": "skipped_no_source"})
            continue
        if dst.exists() and not force:
            records.append({"src": str(src), "dst": str(dst), "status": "skipped_dest_exists"})
            continue

        if dry_run:
            records.append({"src": str(src), "dst": str(dst), "status": "would_copy", "kind": kind})
            continue

        backup_path: str | None = None
        if dst.exists() and force:
            backup_path = str(dst.with_name(f"{dst.name}.bak.pre_phase2_{ts}"))
            shutil.move(str(dst), backup_path)

        if kind == "dir":
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        record: dict[str, Any] = {"src": str(src), "dst": str(dst), "status": "copied", "kind": kind}
        if backup_path:
            record["status"] = "forced_backup_then_copied"
            record["backup"] = backup_path
        records.append(record)

    return records


def ensure_repo_data_layout(*, verbose: bool = True) -> Path:
    """Idempotently create `<repo>/data/` and migrate legacy `~/.jarvis-code/`
    on first run. Subsequent runs short-circuit on the `.migrated` sentinel.

    Returns the resolved data root. Safe to call at every sidecar startup.
    The legacy directory is left intact as a user-side backup.
    """
    data_root = _default_data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    sentinel = data_root / SENTINEL_FILENAME
    if sentinel.exists():
        return data_root

    if LEGACY_HOME_DIR.exists():
        moves = perform_legacy_moves(LEGACY_HOME_DIR, data_root)
    else:
        moves = []

    payload = {
        "migrated_at": datetime.now(UTC).isoformat(),
        "legacy_home": str(LEGACY_HOME_DIR),
        "data_root": str(data_root),
        "legacy_kept": True,
        "moves": moves,
        "note": (
            "One-time Phase 2 isolation migration. Legacy ~/.jarvis-code/ is "
            "kept as a backup. Restart sidecar before relying on the new "
            "paths — module-level singletons cache the old ones. Safe to "
            "delete the legacy directory after end-to-end verification."
        ),
    }
    sentinel.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if verbose:
        copied = [m for m in moves if m.get("status") == "copied"]
        if copied:
            print(
                f"[jarvis:phase2] migrated {len(copied)} legacy item(s) into {data_root}",
                file=sys.stderr,
                flush=True,
            )
            for m in copied:
                print(f"  - {m['src']} -> {m['dst']}", file=sys.stderr, flush=True)
        elif LEGACY_HOME_DIR.exists():
            print(
                f"[jarvis:phase2] {data_root} already populated; legacy at {LEGACY_HOME_DIR} not touched",
                file=sys.stderr,
                flush=True,
            )

    return data_root


def config_path() -> Path:
    raw = os.environ.get("JARVIS_CODE_CONFIG")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    return _default_data_root() / "config.yaml"


def providers_path() -> Path:
    return config_path().with_name("providers.yaml")


def credentials_path() -> Path:
    return config_path().with_name("credentials.yaml")


def internal_memory_root() -> Path:
    configured = os.environ.get("JARVIS_WORKSPACE")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser()
    return _default_data_root() / "workspaceMemory"


def load_sidecar_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_providers_config() -> dict[str, Any]:
    path = providers_path()
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def ensure_sidecar_config() -> dict[str, Any]:
    # Guarantee the user-home data root exists before config reads.
    # JARVIS_WORKSPACE / JARVIS_CODE_CONFIG env overrides still win.
    ensure_repo_data_layout()
    raw = load_sidecar_config()
    changed = False

    for key, value in DEFAULT_ROLE_CONFIG.items():
        if key not in raw or raw.get(key) is None:
            raw[key] = value
            changed = True

    roles = raw.get("roles")
    if not isinstance(roles, dict):
        raw["roles"] = dict(DEFAULT_ROLE_CONFIG["roles"])
        changed = True
    else:
        for role_name, role_value in DEFAULT_ROLE_CONFIG["roles"].items():
            if not isinstance(roles.get(role_name), str) or not str(roles.get(role_name)).strip():
                roles[role_name] = role_value
                changed = True

    if "default_project_root" not in raw or raw.get("default_project_root") is None:
        raw["default_project_root"] = ""
        changed = True

    protected = _normalize_path_list(raw.get("protected_roots"))
    for detected in _detected_protected_roots():
        if detected not in protected:
            protected.append(detected)
            changed = True
    raw["protected_roots"] = protected

    if changed:
        write_sidecar_config(raw)
    ensure_providers_config()
    load_credentials_into_env()
    return raw


def ensure_providers_config() -> dict[str, Any]:
    raw = load_providers_config()
    changed = False
    managed_by_model_setting = raw.get("managed_by") == "jarvis-model-setting"

    defaults = raw.get("defaults")
    if not isinstance(defaults, dict):
        raw["defaults"] = deepcopy(DEFAULT_PROVIDER_CONFIG["defaults"])
        changed = True
    elif not defaults.get("primary"):
        defaults["primary"] = DEFAULT_PROVIDER_CONFIG["defaults"]["primary"]
        changed = True

    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raw["providers"] = {}
        providers = raw["providers"]
        changed = True

    if managed_by_model_setting:
        if "fallback" not in defaults or defaults.get("fallback") is None:
            defaults["fallback"] = []
            changed = True
        if changed:
            write_providers_config(raw)
        return raw

    for provider_name, provider_default in DEFAULT_PROVIDER_CONFIG["providers"].items():
        provider = providers.get(provider_name)
        if not isinstance(provider, dict):
            providers[provider_name] = deepcopy(provider_default)
            changed = True
            continue
        for key, value in provider_default.items():
            if key == "models":
                models = provider.get("models")
                if not isinstance(models, dict):
                    provider["models"] = deepcopy(value)
                    changed = True
                else:
                    for model_name, model_cfg in value.items():
                        if not isinstance(models.get(model_name), dict):
                            models[model_name] = deepcopy(model_cfg)
                            changed = True
                continue
            if key not in provider or provider.get(key) in (None, "", []):
                provider[key] = deepcopy(value)
                changed = True

    if changed:
        write_providers_config(raw)
    return raw


def load_credentials() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def write_credentials(raw: dict[str, Any]) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    _atomic_write_text(path, content)


def load_credentials_into_env() -> dict[str, str]:
    raw = load_credentials()
    env_map = raw.get("env")
    if not isinstance(env_map, dict):
        return {}
    applied: dict[str, str] = {}
    for name, value in env_map.items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        if not name.strip() or not value.strip():
            continue
        os.environ[name] = value
        applied[name] = value
    return applied


def save_credential_env(env_name: str, value: str) -> Path:
    env_name = env_name.strip()
    value = value.strip()
    if not env_name or not value:
        raise ValueError("env_name and value are required")
    raw = load_credentials()
    env_map = raw.setdefault("env", {})
    if not isinstance(env_map, dict):
        raw["env"] = {}
        env_map = raw["env"]
    env_map[env_name] = value
    write_credentials(raw)
    os.environ[env_name] = value
    return credentials_path()


def remove_credential_env(env_name: str) -> Path:
    env_name = env_name.strip()
    if not env_name:
        raise ValueError("env_name is required")
    raw = load_credentials()
    env_map = raw.get("env")
    if isinstance(env_map, dict):
        env_map.pop(env_name, None)
    write_credentials(raw)
    os.environ.pop(env_name, None)
    return credentials_path()


def write_sidecar_config(raw: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    _atomic_write_text(path, content)


def write_providers_config(raw: dict[str, Any]) -> None:
    path = providers_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    _atomic_write_text(path, content)


def setup_default_project_root(path_value: str) -> dict[str, Any]:
    raw = ensure_sidecar_config()
    resolved = _resolve_path_str(path_value)
    if not resolved:
        raise ValueError(f"invalid default_project_root: {path_value!r}")
    Path(resolved).mkdir(parents=True, exist_ok=True)
    raw["default_project_root"] = resolved
    write_sidecar_config(raw)
    return raw


def get_default_project_root() -> str | None:
    raw = ensure_sidecar_config()
    return _resolve_path_str(raw.get("default_project_root"))


def get_effective_project_root() -> str | None:
    configured = get_default_project_root()
    if configured:
        return configured
    return _resolve_path_str(_platform_default_project_root())


def get_project_root_source() -> str:
    return "configured" if get_default_project_root() else "platform_default"


def get_protected_roots() -> list[str]:
    raw = ensure_sidecar_config()
    return _normalize_path_list(raw.get("protected_roots"))


def is_protected_path(path_value: str | None) -> bool:
    resolved = _resolve_path_str(path_value)
    if not resolved:
        return False
    target = Path(resolved)
    for root in get_protected_roots():
        if _is_relative_to(target, Path(root)):
            return True
    return False


def _platform_default_project_root() -> Path:
    override = os.environ.get("JARVIS_DEFAULT_PROJECT_ROOT")
    if isinstance(override, str) and override.strip():
        return Path(override).expanduser()

    home = Path.home()
    if os.name == "nt":
        anchor = home.anchor or ""
        if anchor.strip():
            return Path(anchor) / "jarvis_workspace"
    return home / "jarvis_workspace"


def _detected_protected_roots() -> list[str]:
    sidecar_root = Path(__file__).resolve().parents[1]
    project_root = sidecar_root.parent
    candidates = [sidecar_root, project_root / "pi", _default_data_root()]
    detected: list[str] = []
    for candidate in candidates:
        resolved = _resolve_path_str(candidate)
        if resolved and resolved not in detected:
            detected.append(resolved)
    return detected


def _normalize_path_list(value: Any) -> list[str]:
    items = value if isinstance(value, list) else []
    normalized: list[str] = []
    for item in items:
        resolved = _resolve_path_str(item)
        if resolved and resolved not in normalized:
            normalized.append(resolved)
    return normalized


def _resolve_path_str(path_value: Any) -> str | None:
    if not isinstance(path_value, (str, os.PathLike)):
        return None
    text = str(path_value).strip()
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve())
    except OSError:
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    # Owner-only (0o600) so credential files (credentials.yaml holds provider API
    # keys) are not world-readable on shared POSIX hosts; best-effort on Windows.
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
