# Memory and Projects

JARVIS Code is designed around project memory rather than long raw prompt
replay.

## Chat Mode

Chat is the default entry mode. The model first interprets the user's intent
inside the normal chat turn. It may remain in chat, route to unregistered coding,
or move into deep project work.

## Project Memory

The sidecar resolves project context and returns a compact JLC memory block for
the current user request. That block is injected into the latest user message as
`<jarvis_memory>`.

Old turns are kept in local raw storage for retrieval, not sent directly as a
large prefix.

## Deepdive Modes

JARVIS Code supports progressively heavier project modes:

- chat
- unregistered coding
- deepdive
- heavy deepdive

The active mode controls project memory, thinking level, and how aggressively
JARVIS should inspect and modify a project.

## Runtime History Limit

The internal live runtime keeps a bounded recent-turn fallback:

```text
JARVIS_RUNTIME_HISTORY_TURNS=100
```

This is not the model payload policy. Provider requests are still trimmed by the
JARVIS extension so old raw prefixes are not sent when JLC context is available
or when the latest user turn can be identified.

## Subturn Payload Limit

Long tool loops inside one user request are also bounded:

```text
JARVIS_SUBTURN_HISTORY_MESSAGES=100
```

If unset, JARVIS uses `JARVIS_RUNTIME_HISTORY_TURNS` as the subturn payload item
limit. Trimming keeps the latest complete tool/function-call groups so provider
payloads do not grow unbounded while compaction remains disabled.
