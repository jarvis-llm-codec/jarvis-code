---
description: Main implements, reviewer reviews, main applies feedback
---
Use this workflow when implementation should receive an independent review:

1. Decide whether subagents are actually needed. For small or obvious edits, do not use subagents.
2. If needed, use "scout" before editing to gather focused context for: $@
3. The main agent implements the change and runs appropriate validation.
4. Use "reviewer" to review the actual diff, changed files, and validation results.
5. The main agent decides which reviewer findings to apply, makes any fixes, reruns validation, and owns the final response.

Do not use "worker" automatically. Use worker only for isolated new files, docs, tests, examples, or clearly separated modules.
