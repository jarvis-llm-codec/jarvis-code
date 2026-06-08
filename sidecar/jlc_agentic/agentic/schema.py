"""Tool schemas and dispatcher wiring."""
from __future__ import annotations

from jlc_agentic.providers import get_llm

from . import subagent as _subagent
from .dispatcher import ToolDispatcher
from .tools import (
    bash,
    edit,
    grep,
    jre_search,
    read,
    recall_turn,
    register_project,
    switch_project,
    web_search,
    write_file,
)

_BASE_SCHEMAS = [
    read.SCHEMA,
    grep.SCHEMA,
    edit.SCHEMA,
    write_file.SCHEMA,
    bash.SCHEMA,
    web_search.SCHEMA,
    recall_turn.SCHEMA,
    jre_search.SCHEMA,
    register_project.SCHEMA,
    switch_project.SCHEMA,
]

_BASE_HANDLERS = {
    "read": read.handler,
    "grep": grep.handler,
    "edit": edit.handler,
    "write_file": write_file.handler,
    "bash": bash.handler,
    "web_search": web_search.handler,
    "recall_turns": recall_turn.handler,
    "jre_search": jre_search.handler,
    "register_project": register_project.handler,
    "switch_project": switch_project.handler,
}

ALL_TOOLS = [*_BASE_SCHEMAS, _subagent.SCHEMA]
SUBAGENT_TOOLS = list(_BASE_SCHEMAS)


def get_dispatcher(
    subagent_llm_client: object | None = None,
    subagent_on_token: object | None = None,
    conv_id: str | None = None,
    storage_root: str | None = None,
    project_root: str | None = None,
    on_external_write: object | None = None,
    retriever: object | None = None,
) -> ToolDispatcher:
    """Build the main-agent dispatcher (includes delegate_subagent).

    subagent_on_token (optional): forwarded to make_handler so the
    subagent's reasoning/content tokens stream live to the caller's
    terminal instead of being buffered until the summary returns.

    conv_id / storage_root (optional): when supplied, the recall_turns
    handler is wrapped in a closure that injects them on every call so
    the LLM does not have to (and cannot) pick the wrong memory store.
    Both must point at the same JHB root the host JarvisAgentic writes to.

    retriever (optional): host's singleton JLCRetriever. When supplied,
    recall_turns reuses it instead of building a fresh one per call —
    avoids re-loading the sentence-transformer weights on every recall.

    on_external_write (optional): callback fired by ToolDispatcher when
    write_file lands outside active_project_path. Used as a safety net for
    the [New project bootstrap] directive — host can auto-register the
    new folder so the LLM forgetting register_project does not strand the
    session.
    """
    effective = subagent_llm_client if subagent_llm_client is not None else get_llm("subagent")
    delegate_handler = _subagent.make_handler(
        llm_client=effective,
        on_token=subagent_on_token,
        conv_id=conv_id,
        storage_root=storage_root,
        project_root=project_root,
        retriever=retriever,
    )
    handlers = _bind_recall_turn(dict(_BASE_HANDLERS), conv_id, storage_root, retriever)
    handlers = _bind_path_tools(handlers, project_root)
    handlers["delegate_subagent"] = delegate_handler
    return ToolDispatcher(
        handlers,
        active_project_path=project_root,
        on_external_write=on_external_write,
    )


def get_subagent_dispatcher(
    conv_id: str | None = None,
    storage_root: str | None = None,
    project_root: str | None = None,
    retriever: object | None = None,
) -> ToolDispatcher:
    """Build a dispatcher for use INSIDE a subagent.

    Accepts the same conv_id/storage_root/retriever binding as
    get_dispatcher so delegated subagents read from the same JHB store
    the host writes to AND share the same warm sentence-transformer.
    Without this, recall_turns inside a subagent silently falls back to
    the config default and rebuilds the retriever (cold model load).
    """
    handlers = _bind_recall_turn(dict(_BASE_HANDLERS), conv_id, storage_root, retriever)
    handlers = _bind_path_tools(handlers, project_root)
    return ToolDispatcher(handlers, active_project_path=project_root)


def _bind_recall_turn(
    handlers: dict,
    conv_id: str | None,
    storage_root: str | None,
    retriever: object | None = None,
) -> dict:
    """Wrap recall_turns in a closure that injects conv_id/storage_root/retriever
    and swallows any LLM-supplied overrides. No-op when all three are None
    so direct callers (tests, ad-hoc usage) keep the config-default fallback.
    """
    if conv_id is None and storage_root is None and retriever is None:
        return handlers
    base_recall = handlers["recall_turns"]
    bound_conv = conv_id if conv_id is not None else "conversation"
    bound_root = storage_root
    bound_retriever = retriever

    def _recall_bound(
        queries: list | str | None = None,
        top_k: int = 5,
        query: str | None = None,
        **_ignored: object,
    ) -> dict:
        if _ignored:
            # Angle 3 jailbreak telemetry — LLM-supplied conv_id/storage_root
            # /retriever overrides are silently swallowed by the closure;
            # log them so injection attempts are visible in stderr without
            # changing behavior. Keep keys only (values may be sensitive).
            import sys
            sys.stderr.write(
                f"[recall_turns:override-attempt] swallowed extra kwargs "
                f"{sorted(_ignored.keys())}\n"
            )
        return base_recall(
            queries=queries,
            query=query,
            top_k=top_k,
            conv_id=bound_conv,
            storage_root=bound_root,
            retriever=bound_retriever,
        )

    handlers["recall_turns"] = _recall_bound
    return handlers



def _bind_path_tools(
    handlers: dict,
    project_root: str | None,
) -> dict:
    """Optional initial bind for path-aware tools.

    With Option A, dispatcher.active_project_path is mutable and updated per
    chat turn. This helper only seeds an initial root for direct callers and
    remains a no-op when project_root is None.
    """
    if project_root is None:
        return handlers
    path_tools = {"read", "edit", "write_file", "grep", "bash"}
    for name in path_tools:
        base = handlers.get(name)
        if base is None:
            continue

        def _bound(*_args, _base=base, **kwargs):
            kwargs["project_root"] = project_root
            return _base(**kwargs)

        handlers[name] = _bound
    return handlers
