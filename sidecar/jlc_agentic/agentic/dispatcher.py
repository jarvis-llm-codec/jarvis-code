"""Tool dispatcher for agentic loop."""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

PATH_AWARE_TOOLS = {"read", "edit", "write_file", "grep", "bash"}
PROJECT_ROOT_TOOLS = PATH_AWARE_TOOLS | {"delegate_subagent"}

READ_ONLY_TOOLS = {"read", "grep", "web_search", "recall_turns", "jre_search"}
SEQUENTIAL_TOOLS = {"edit", "bash"}
MAX_INLINE_BYTES = 50 * 1024


class ToolDispatcher:
    """Dispatch named tool calls to registered handlers."""

    def __init__(
        self,
        tools: dict[str, Callable[..., dict[str, Any]]],
        active_project_path: str | None = None,
        on_external_write: Callable[[str], None] | None = None,
    ) -> None:
        self.tools = tools
        self.active_project_path = active_project_path
        # Optional safety-net callback: fired after a successful write_file
        # whose absolute path falls OUTSIDE active_project_path. The host
        # (JlcAgenticCoder) uses it to auto-register the new folder so the
        # session that created it does not become unrouteable. Plain Callable
        # — None disables the hook.
        self.on_external_write = on_external_write

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool and normalize result shape."""
        handler = self.tools.get(name)
        if handler is None:
            return {"ok": False, "result": None, "error": f"unknown tool: {name}"}
        try:
            call_args = dict(args or {})
            if name in PROJECT_ROOT_TOOLS and self.active_project_path is not None:
                call_args["project_root"] = self.active_project_path
            result = handler(**call_args)
            packed = {"ok": True, "result": result, "error": None}
            self._maybe_fire_external_write(name, call_args, result)
            return self._maybe_spill(packed, name)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "result": None, "error": str(exc)}

    def _maybe_fire_external_write(
        self,
        name: str,
        call_args: dict[str, Any],
        result: dict[str, Any] | Any,
    ) -> None:
        """Fire on_external_write when write_file landed outside active root.

        Heuristic safety net for the directive in slim.py [New project
        bootstrap]: if the LLM forgets to call register_project after
        writing into a brand-new sibling folder, the host gets a callback
        with the absolute folder path so it can auto-register. Failures
        are swallowed — a broken hook must NOT break the tool call.
        """
        if self.on_external_write is None or name != "write_file":
            return
        try:
            written = (result or {}).get("path") if isinstance(result, dict) else None
            if not written:
                return
            written_path = Path(written)
            if not written_path.is_absolute():
                return
            if self.active_project_path:
                try:
                    written_path.resolve().relative_to(Path(self.active_project_path).resolve())
                    return  # inside active root — nothing to do
                except (ValueError, OSError):
                    pass
            # write_file always writes a file, so the project candidate is
            # the file's parent directory.
            candidate = written_path.parent
            if not candidate.exists() or not candidate.is_dir():
                return
            self.on_external_write(str(candidate))
        except Exception:
            return

    def execute_all(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute calls; read-only calls run in parallel, write calls are sequential."""
        results: list[dict[str, Any] | None] = [None] * len(tool_calls)

        def run_block(indices: list[int]) -> None:
            if not indices:
                return
            with ThreadPoolExecutor(max_workers=len(indices)) as pool:
                futures = {
                    pool.submit(self.execute, tool_calls[i]["name"], tool_calls[i].get("args", {})): i
                    for i in indices
                }
                for fut, idx in futures.items():
                    results[idx] = fut.result()

        read_block: list[int] = []
        for idx, call in enumerate(tool_calls):
            name = call["name"]
            if name in READ_ONLY_TOOLS:
                read_block.append(idx)
                continue
            run_block(read_block)
            read_block = []
            results[idx] = self.execute(name, call.get("args", {}))
        run_block(read_block)
        return [r if r is not None else {"ok": False, "result": None, "error": "execution failed"} for r in results]

    @staticmethod
    def _maybe_spill(payload: dict[str, Any], tool_name: str) -> dict[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(raw) <= MAX_INLINE_BYTES:
            return payload
        path = ToolDispatcher._spill_to_file(raw, tool_name)
        return {
            "ok": True,
            "result": {
                "spilled": True,
                "path": str(path),
                "bytes": len(raw),
                "note": "Result exceeded 50KB and was saved to file.",
            },
            "error": None,
        }

    @staticmethod
    def _spill_to_file(raw: bytes, tool_name: str) -> Path:
        fd, tmp_name = tempfile.mkstemp(prefix=f"jlc_tool_{tool_name}_", suffix=".json")
        try:
            import os

            os.close(fd)
        except Exception:
            pass
        with Path(tmp_name).open("wb") as fh:
            fh.write(raw)
        try:
            Path(tmp_name).chmod(0o600)
        except Exception:
            pass
        return Path(tmp_name)
