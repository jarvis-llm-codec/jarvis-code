#!/usr/bin/env python3
"""JARVIS Code installation diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PI_ROOT = ROOT / "pi"
SIDECAR_ROOT = ROOT / "sidecar"
SIDECAR_VENV = SIDECAR_ROOT / ".venv"
PI_AGENT_DIR = ROOT / "pi-agent"
DATA_DIR = ROOT / "data"
EMBEDDER_MODEL = "BAAI/bge-m3"
WINDOWS_VC_REDIST_DLLS = ("vcruntime140_1.dll", "msvcp140.dll")
WINDOWS_VC_REDIST_MESSAGE = (
    "Microsoft Visual C++ Redistributable (x64) is required for the memory/embedding layer: "
    "https://aka.ms/vs/17/release/vc_redist.x64.exe"
)


def config_summary_line() -> str:
    env_state = "set" if os.environ.get("JARVIS_CODE_CONFIG") else "unset"
    try:
        if str(SIDECAR_ROOT) not in sys.path:
            sys.path.insert(0, str(SIDECAR_ROOT))
        from jarvis_sidecar.config import config_path

        return f"config: {config_path().resolve()} (env JARVIS_CODE_CONFIG {env_state})"
    except Exception as exc:
        return f"config: unavailable ({type(exc).__name__}: {exc}; env JARVIS_CODE_CONFIG {env_state})"


def provider_catalog_summary_line() -> str:
    try:
        if str(SIDECAR_ROOT) not in sys.path:
            sys.path.insert(0, str(SIDECAR_ROOT))
        from jarvis_sidecar.llm_setting import catalog_overlay_summary

        summary = catalog_overlay_summary()
        user_path = summary["user_path"]
        if not summary["user_exists"]:
            return f"user overlay: none ({user_path})"
        return f"providers: catalog {summary['repo_count']} + user {summary['user_count']} ({user_path})"
    except Exception as exc:
        return f"providers: unavailable ({type(exc).__name__}: {exc})"


def check_config_and_provider_catalog(checks: list[Check]) -> None:
    try:
        if str(SIDECAR_ROOT) not in sys.path:
            sys.path.insert(0, str(SIDECAR_ROOT))
        from jarvis_sidecar.config import config_path

        add(checks, "config", "ok", str(config_path().resolve()))
    except Exception as exc:
        add(checks, "config", "warn", f"{type(exc).__name__}: {exc}")

    try:
        if str(SIDECAR_ROOT) not in sys.path:
            sys.path.insert(0, str(SIDECAR_ROOT))
        from jarvis_sidecar.llm_setting import catalog_overlay_summary

        summary = catalog_overlay_summary()
        if summary["user_exists"]:
            message = f"repo={summary['repo_count']} user={summary['user_count']} path={summary['user_path']}"
        else:
            message = f"repo={summary['repo_count']} user=0 path={summary['user_path']}"
        add(checks, "providers", "ok", message)
    except Exception as exc:
        add(checks, "providers", "warn", f"{type(exc).__name__}: {exc}")


def platform_summary_line() -> str:
    return f"platform: {platform.platform()} ({platform.machine()}); python={sys.version.split()[0]} at {sys.executable}"


def configure_hf_public_download_env() -> None:
    has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not has_hf_token:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_UPDATE_CHECK", "1")


@dataclass
class Check:
    name: str
    status: str
    message: str


def is_windows() -> bool:
    return os.name == "nt"


def vc_redist_x64_installed() -> bool:
    if not is_windows():
        return True
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
    system32 = Path(system_root) / "System32"
    return all((system32 / dll).exists() for dll in WINDOWS_VC_REDIST_DLLS)


def venv_python() -> Path:
    if is_windows():
        return SIDECAR_VENV / "Scripts" / "python.exe"
    return SIDECAR_VENV / "bin" / "python"


def tsx_path() -> Path:
    if is_windows():
        return PI_ROOT / "node_modules" / ".bin" / "tsx.cmd"
    return PI_ROOT / "node_modules" / ".bin" / "tsx"


def add(checks: list[Check], name: str, status: str, message: str) -> None:
    checks.append(Check(name=name, status=status, message=message))


def run_capture(command: list[str], *, cwd: Path | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or f"timeout after {timeout}s"
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def command_version(command: str, args: list[str]) -> tuple[bool, str]:
    found = shutil.which(command)
    if not found:
        return False, f"{command} not found"
    code, out, err = run_capture([found, *args])
    if code != 0:
        return False, err or out or f"{command} exited {code}"
    return True, out or found


def check_node(checks: list[Check]) -> None:
    ok, text = command_version("node", ["--version"])
    if not ok:
        add(checks, "node", "fail", text)
        return
    major_text = text.lstrip("v").split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError:
        add(checks, "node", "fail", f"could not parse Node version: {text}")
        return
    if major < 20:
        add(checks, "node", "fail", f"{text}; Node.js 20+ required")
    else:
        add(checks, "node", "ok", text)


def check_command(checks: list[Check], name: str, command: str, args: list[str]) -> None:
    ok, text = command_version(command, args)
    summary = text.splitlines()[0] if text.splitlines() else text
    add(checks, name, "ok" if ok else "fail", summary)


def check_windows_vc_redist(checks: list[Check]) -> None:
    if not is_windows():
        return
    if vc_redist_x64_installed():
        add(checks, "windows:vcredist-x64", "ok", "Microsoft Visual C++ 2015-2022 Redistributable (x64)")
        return
    add(checks, "windows:vcredist-x64", "fail", WINDOWS_VC_REDIST_MESSAGE)


def check_platform(checks: list[Check]) -> None:
    add(checks, "platform", "ok", f"{platform.system()} {platform.release()} ({platform.machine()})")


def check_posix_install_tools(checks: list[Check]) -> None:
    if is_windows():
        return
    check_command(checks, "curl", "curl", ["--version"])
    check_command(checks, "tar", "tar", ["--version"])


def check_paths(checks: list[Check]) -> None:
    required = {
        "root": ROOT,
        "pi": PI_ROOT,
        "sidecar": SIDECAR_ROOT,
        "pi-agent": PI_AGENT_DIR,
        "data": DATA_DIR,
    }
    for name, path in required.items():
        add(checks, f"path:{name}", "ok" if path.exists() else "fail", str(path))
    tsx = tsx_path()
    add(checks, "node:tsx", "ok" if tsx.exists() else "fail", str(tsx))


def check_python_venv(checks: list[Check]) -> Path | None:
    py = venv_python()
    if not py.exists():
        add(checks, "python:venv", "fail", f"sidecar venv python not found at {py}")
        return None
    code, out, err = run_capture([str(py), "-c", "import sys; print(sys.version.split()[0])"])
    if code != 0:
        add(checks, "python:venv", "fail", err or out or f"python exited {code}")
        return None
    add(checks, "python:venv", "ok", f"{out} at {py}")
    return py


def check_python_packages(checks: list[Check], py: Path | None) -> None:
    if py is None:
        return
    modules = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pyyaml", "yaml"),
        ("httpx", "httpx"),
        ("tiktoken", "tiktoken"),
        ("litellm", "litellm"),
        ("sentence-transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("rank-bm25", "rank_bm25"),
    ]
    code = (
        "import importlib, json\n"
        f"mods = {json.dumps(modules)}\n"
        "out = {}\n"
        "for label, name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        out[label] = 'ok'\n"
        "    except Exception as exc:\n"
        "        out[label] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(out, ensure_ascii=False))\n"
    )
    exit_code, out, err = run_capture([str(py), "-c", code], timeout=60)
    if exit_code != 0:
        add(checks, "python:packages", "fail", err or out or f"python exited {exit_code}")
        return
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        add(checks, "python:packages", "fail", out)
        return
    for label, status in result.items():
        add(checks, f"python:{label}", "ok" if status == "ok" else "fail", status)


def check_embedder(checks: list[Check], py: Path | None, *, preload: bool, require: bool, skip_load: bool) -> None:
    if py is None:
        return
    if skip_load and not preload:
        add(checks, "embedder:bge-m3", "warn", "not checked; run `jarvis doctor --preload-embedder` to verify/download")
        return

    local_only = "False" if preload else "True"
    encode = "True" if preload else "False"
    code = f"""
import json
import sys
from huggingface_hub import snapshot_download
model = {EMBEDDER_MODEL!r}
try:
    path = snapshot_download(repo_id=model, local_files_only={local_only})
    loaded = False
    if {encode}:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model, device='cpu')
        _ = m.encode(['warmup'], normalize_embeddings=True, show_progress_bar=False)
        loaded = True
    print(json.dumps({{'ok': True, 'path': str(path), 'loaded': loaded}}))
except Exception as exc:
    print(json.dumps({{'ok': False, 'error': f'{{type(exc).__name__}}: {{exc}}'}}))
    sys.exit(1)
"""
    exit_code, out, err = run_capture([str(py), "-c", code], timeout=1800)
    status = "ok" if exit_code == 0 else ("fail" if require else "warn")
    if exit_code == 0:
        try:
            payload = json.loads(out)
            message = f"{EMBEDDER_MODEL} cached at {payload.get('path')}"
            if payload.get("loaded"):
                message += "; load+encode ok"
        except json.JSONDecodeError:
            message = out
    else:
        message = ""
        try:
            payload = json.loads(out)
            if isinstance(payload, dict):
                message = str(payload.get("error") or "")
        except json.JSONDecodeError:
            pass
        if not message:
            message = out.strip()
        if err:
            # Hugging Face progress bars and transformers warnings use stderr.
            # Keep only the tail so the actual error remains visible.
            err_tail = "\n".join(err.splitlines()[-12:]).strip()
            if err_tail:
                message = f"{message}\n{err_tail}".strip()
        if not message:
            message = f"{EMBEDDER_MODEL} unavailable"
    add(checks, "embedder:bge-m3", status, message)


def _path_is_under(child: str | None, parent: Path) -> bool:
    if not child:
        return False
    try:
        child_path = Path(child).resolve()
        parent_path = parent.resolve()
        child_path.relative_to(parent_path)
        return True
    except (OSError, ValueError):
        return False


def check_sidecar(checks: list[Check]) -> None:
    url = os.environ.get("JARVIS_SIDECAR_URL", "http://127.0.0.1:8765")
    status_url = f"{url.rstrip('/')}/status"
    try:
        with urllib.request.urlopen(status_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        add(checks, "sidecar", "warn", f"not running or not reachable: {exc}")
        return
    except json.JSONDecodeError as exc:
        add(checks, "sidecar", "warn", f"invalid status response: {exc}")
        return
    if payload.get("ok") is False:
        add(checks, "sidecar", "warn", json.dumps(payload, ensure_ascii=False)[:500])
        return
    agent_loaded = payload.get("agent_loaded")
    last_error = payload.get("last_agent_error")
    app_file = str(payload.get("sidecar_app_file") or "")
    python_executable = str(payload.get("python_executable") or "")
    process_id = payload.get("process_id")
    next_retry_at = payload.get("next_retry_at")
    details = []
    if process_id:
        details.append(f"pid={process_id}")
    if app_file:
        details.append(f"app={app_file}")
    if python_executable:
        details.append(f"python={python_executable}")
    if not _path_is_under(app_file, SIDECAR_ROOT):
        details.append(f"expected_sidecar={SIDECAR_ROOT}")
    if agent_loaded is False or last_error:
        error_parts = [f"agent_loaded={agent_loaded}", f"last_agent_error={last_error}"]
        error_type = payload.get("last_agent_error_type")
        error_repr = payload.get("last_agent_error_repr")
        error_filename = payload.get("last_agent_error_filename")
        if error_type:
            error_parts.append(f"type={error_type}")
        if error_filename:
            error_parts.append(f"filename={error_filename}")
        if error_repr and error_repr != last_error:
            error_parts.append(f"repr={error_repr}")
        if next_retry_at:
            error_parts.append(f"next_retry_at={next_retry_at}")
        if details:
            error_parts.append("; ".join(details))
        error_parts.append("If this follows install/update, run `jarvis` once to restart the sidecar.")
        add(checks, "sidecar", "warn", "; ".join(str(part) for part in error_parts if part))
    else:
        message = status_url
        if details:
            message = f"{message}; {'; '.join(details)}"
        add(checks, "sidecar", "ok", message)


def print_text(checks: list[Check]) -> None:
    labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    print(platform_summary_line())
    print(config_summary_line())
    print(provider_catalog_summary_line())
    print("JARVIS Code doctor")
    for check in checks:
        label = labels.get(check.status, check.status.upper())
        print(f"[{label}] {check.name}: {check.message}")


def main() -> int:
    configure_hf_public_download_env()
    parser = argparse.ArgumentParser(prog="jarvis doctor")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--preload-embedder", action="store_true", help="download and load BAAI/bge-m3")
    parser.add_argument("--require-embedder", action="store_true", help="treat missing bge-m3 as a failure")
    parser.add_argument("--skip-embedder-load", action="store_true", help="skip local bge-m3 cache check")
    parser.add_argument("--skip-sidecar", action="store_true", help="skip sidecar status probe")
    args = parser.parse_args()

    checks: list[Check] = []
    check_platform(checks)
    check_paths(checks)
    check_config_and_provider_catalog(checks)
    check_command(checks, "git", "git", ["--version"])
    check_posix_install_tools(checks)
    check_node(checks)
    check_command(checks, "npm", "npm", ["--version"])
    check_windows_vc_redist(checks)
    py = check_python_venv(checks)
    check_python_packages(checks, py)
    check_embedder(
        checks,
        py,
        preload=args.preload_embedder,
        require=args.require_embedder or args.preload_embedder,
        skip_load=args.skip_embedder_load,
    )
    if not args.skip_sidecar:
        check_sidecar(checks)

    if args.json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        print_text(checks)
    return 1 if any(check.status == "fail" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
