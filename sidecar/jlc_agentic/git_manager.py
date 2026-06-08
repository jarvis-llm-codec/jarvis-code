"""Git helper for jhb history management."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_SESSION_ID = "jarvis_session"


class GitManager:
    def __init__(self, jhb_root: Path) -> None:
        self.jhb_root = Path(jhb_root)
        self._git = shutil.which("git")
        self._available = self._git is not None

    def init_repo(self) -> None:
        self.jhb_root.mkdir(parents=True, exist_ok=True)
        if not self._available:
            print("[jlc:git] git not found; JHB git history disabled", file=sys.stderr)
            return
        if (self.jhb_root / ".git").exists():
            return

        try:
            self._run("init")
            self._run("config", "user.name", "jlc-bot")
            self._run("config", "user.email", "jlc-bot@example.local")
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
            self._available = False
            print(f"[jlc:git] init failed; JHB git history disabled: {exc}", file=sys.stderr)

    def auto_commit(self, turn: int, session_id: str | None = None) -> str | None:
        """Commit jhb changes if tracked file changed. Never raises — returns None on failure.

        Self-heals once if .git is gone (e.g., manual reset). All other errors log and return None
        so downstream persistence (retriever, etc.) is never aborted by git problems.
        """
        if not self._available:
            return None
        try:
            return self._auto_commit_impl(turn, session_id=session_id)
        except subprocess.CalledProcessError as exc:
            if not (self.jhb_root / ".git").exists():
                try:
                    self.init_repo()
                    return self._auto_commit_impl(turn, session_id=session_id)
                except Exception as heal_exc:  # noqa: BLE001
                    print(
                        f"[jlc:git] auto_commit self-heal failed: {heal_exc}",
                        file=sys.stderr,
                    )
                    return None
            print(f"[jlc:git] auto_commit failed (CalledProcessError): {exc}", file=sys.stderr)
            return None
        except (FileNotFoundError, OSError) as exc:
            print(f"[jlc:git] auto_commit failed (I/O): {exc}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"[jlc:git] auto_commit failed (unexpected {type(exc).__name__}): {exc}", file=sys.stderr)
            return None

    def _auto_commit_impl(self, turn: int, session_id: str | None = None) -> str | None:
        safe = self._sanitize_conv_id(session_id)
        rel_path = f"{safe}/jhb.md"
        full_path = self.jhb_root / rel_path
        if not full_path.exists():
            return None

        self._run("add", rel_path)

        staged = self._run("diff", "--cached", "--quiet", check=False)
        if staged.returncode == 0:
            return None

        self._run("commit", "-m", f"jlc: update jhb {safe} turn {turn}")
        head = self._run("rev-parse", "--short", "HEAD")
        return head.stdout.strip() or None

    def get_history(self, limit: int = 20, session_id: str | None = None) -> list[dict]:
        safe = self._sanitize_conv_id(session_id)
        if not self._available:
            return []
        target = self.jhb_root / safe
        if not target.exists():
            return []

        result = self._run(
            "log",
            f"-n{limit}",
            "--date=iso",
            "--pretty=format:%h%x09%ad%x09%s",
            "--",
            f"{safe}/",
            check=False,
        )
        if result.returncode != 0:
            return []

        commits: list[dict] = []
        for line in result.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            commits.append({"hash": parts[0], "date": parts[1], "message": parts[2]})
        return commits

    def rollback(self, commit_hash: str, session_id: str | None = None) -> bool:
        safe = self._sanitize_conv_id(session_id)
        if not self._available:
            return False
        rel_path = f"{safe}/jhb.md"
        target = self.jhb_root / rel_path
        if not target.exists():
            return False

        result = self._run("checkout", commit_hash, "--", rel_path, check=False)
        return result.returncode == 0

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        if not self._git:
            raise FileNotFoundError("git")
        return subprocess.run(
            [self._git, "-C", str(self.jhb_root), *args],
            capture_output=True,
            text=True,
            check=check,
        )

    @staticmethod
    def _sanitize_conv_id(session_id: str | None = None) -> str:
        raw = str(session_id or _SESSION_ID).strip() or _SESSION_ID
        invalid = '<>:"/\\|?*'
        table = str.maketrans({ch: "_" for ch in invalid})
        return raw.translate(table).replace("..", "_")

