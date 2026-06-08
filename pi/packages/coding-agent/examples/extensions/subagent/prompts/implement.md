---
description: Conservative implementation workflow - subagents assist, main implements
---
Use this workflow for non-trivial implementation work:

1. Decide whether subagents are actually needed. For simple or obvious changes, do not use subagents.
2. If the relevant code is unfamiliar, multi-file, or risky, use "scout" to gather focused context for: $@
3. If the implementation has meaningful design choices, use "planner" to create a concrete plan from the scout output and requirements.
4. The main agent implements the change, validates it, and owns the final result.
5. Do not use "worker" unless the implementation is isolated to new files, docs, tests, examples, or clearly separated modules.

Prefer sequential handoffs. Avoid parallel subagents unless the investigations are independent and read-only.
