"""ChatGPT OAuth /responses endpoint adapter.

chatgpt.com/backend-api/codex only accepts OpenAI Responses API (`/responses`),
not chat completions. This module bridges the schemas:

- chat completions messages -> Responses API `input` items
- ResponsesAPIResponse / stream events -> ChatCompletion-compatible ModelResponse

The HTTP call is made by ProviderRouter; this module only handles schema
mapping so it does not require the optional LiteLLM/Aider runtime.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from jlc_agentic.user_agent import jarvis_code_user_agent

def chat_tools_to_responses_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert Chat Completions function tools to Responses API tools."""
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            converted.append(dict(tool))
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            name = tool.get("name")
            if name:
                converted.append(dict(tool))
            continue
        name = fn.get("name")
        if not name:
            continue
        item = {
            "type": "function",
            "name": name,
            "parameters": fn.get("parameters") or {
                "type": "object",
                "properties": {},
            },
            # Chat Completions tools in this repo are not guaranteed to satisfy
            # strict Responses structured-output constraints. Keep validation at
            # the tool dispatcher boundary instead of risking API rejection.
            "strict": bool(fn.get("strict", False)),
        }
        if fn.get("description"):
            item["description"] = fn["description"]
        converted.append(item)
    return converted


def chat_messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant" and msg.get("tool_calls"):
            if content:
                items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": str(content)}],
                })
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = fn.get("name")
                if not name:
                    continue
                call_id = str(call.get("id") or call.get("call_id") or name)
                arguments = fn.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    import json as _json
                    arguments = _json.dumps(arguments, ensure_ascii=False)
                items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": str(name),
                    "arguments": arguments,
                    "status": "completed",
                })
            continue
        if role == "tool":
            output = content if isinstance(content, str) else str(content)
            call_id = str(msg.get("tool_call_id") or msg.get("call_id") or msg.get("name") or "tool_call")
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
                "status": "completed",
            })
            continue
        if content is None:
            continue
        # Responses API requires assistant turns to use output_text, not input_text;
        # mixing input_text under role=assistant returns 400 on multi-turn history.
        text_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            if not content:
                continue
            content_arr = [{"type": text_type, "text": content}]
        elif isinstance(content, list):
            if not content:
                continue
            content_arr = content
        else:
            content_arr = [{"type": text_type, "text": str(content)}]
        items.append({"role": role, "content": content_arr})
    return items


_DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful coding assistant."


def split_instructions_and_input(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split chat messages into (instructions, non-system messages).

    ChatGPT Codex /responses requires non-empty `instructions`. We extract
    system-role messages and concatenate them; if none, fall back to a
    minimal default so the request passes validation.
    """
    instruction_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    instruction_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (
                        "input_text",
                        "text",
                    ):
                        text = block.get("text", "")
                        if text:
                            instruction_parts.append(text)
        else:
            rest.append(msg)
    instructions = "\n\n".join(instruction_parts) or _DEFAULT_CODEX_INSTRUCTIONS
    return instructions, rest


def _extract_text_from_output(output: Any) -> str:
    if not output:
        return ""
    parts: list[str] = []
    for item in output:
        item_type = _attr_or_key(item, "type")
        if item_type != "message":
            continue
        item_content = _attr_or_key(item, "content") or []
        for block in item_content:
            block_type = _attr_or_key(block, "type")
            if block_type in ("output_text", "text"):
                text = _attr_or_key(block, "text") or ""
                parts.append(str(text))
    return "".join(parts)


def _attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _usage_value(usage_obj: Any) -> SimpleNamespace:
    prompt = int(_attr_or_key(usage_obj, "input_tokens") or 0)
    completion = int(_attr_or_key(usage_obj, "output_tokens") or 0)
    total = int(_attr_or_key(usage_obj, "total_tokens") or (prompt + completion))
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _usage_dict(usage_obj: Any) -> dict[str, Any]:
    if not usage_obj:
        return {}
    result = {
        "input_tokens": int(_attr_or_key(usage_obj, "input_tokens") or 0),
        "output_tokens": int(_attr_or_key(usage_obj, "output_tokens") or 0),
        "total_tokens": int(_attr_or_key(usage_obj, "total_tokens") or 0),
    }
    input_details = _attr_or_key(usage_obj, "input_tokens_details")
    if input_details is not None:
        result["input_tokens_details"] = {
            "cached_tokens": int(_attr_or_key(input_details, "cached_tokens") or 0),
        }
    output_details = _attr_or_key(usage_obj, "output_tokens_details")
    if output_details is not None:
        result["output_tokens_details"] = {
            "reasoning_tokens": int(_attr_or_key(output_details, "reasoning_tokens") or 0),
        }
    return result


def _chat_completion_response(
    *,
    response_id: str,
    model: str,
    content: str,
    finish_reason: str,
    usage_obj: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        model=model,
        usage=_usage_value(usage_obj),
    )


def responses_to_chat_completion(resp: Any, model: str) -> SimpleNamespace:
    output_text = _attr_or_key(resp, "output_text")
    if not output_text:
        output_text = _extract_text_from_output(_attr_or_key(resp, "output"))

    status = _attr_or_key(resp, "status") or "completed"
    finish_reason = "stop" if status in ("completed", "succeeded") else status
    return _chat_completion_response(
        response_id=str(_attr_or_key(resp, "id") or ""),
        model=str(_attr_or_key(resp, "model") or model),
        content=str(output_text or ""),
        finish_reason=str(finish_reason),
        usage_obj=_attr_or_key(resp, "usage"),
    )


def consume_responses_stream(stream: Any, model: str) -> SimpleNamespace:
    """Drain a SyncResponsesAPIStreamingIterator and wrap as ChatCompletion.

    Streaming callers receive events. We accumulate `output_text.delta`
    events and pull usage from the terminal `response.completed` or
    `response.incomplete` event.
    """
    text_parts: list[str] = []
    usage_obj: Any = None
    finish_reason = "stop"
    response_id = ""
    response_model = model

    try:
        for event in stream:
            event_type = _attr_or_key(event, "type")
            if os.environ.get("JLC_CODEX_RESPONSES_DEBUG") == "1":
                print(f"[codex-responses-event] {event_type}", flush=True)
            if event_type == "response.output_text.delta":
                delta = _attr_or_key(event, "delta") or ""
                if delta:
                    text_parts.append(str(delta))
            elif event_type == "response.completed":
                resp_obj = _attr_or_key(event, "response")
                usage_obj = _attr_or_key(resp_obj, "usage")
                response_id = str(_attr_or_key(resp_obj, "id") or "")
                response_model = str(_attr_or_key(resp_obj, "model") or model)
                finish_reason = "stop"
            elif event_type == "response.incomplete":
                resp_obj = _attr_or_key(event, "response")
                usage_obj = _attr_or_key(resp_obj, "usage")
                response_id = str(_attr_or_key(resp_obj, "id") or "")
                finish_reason = "length"
            elif event_type in ("response.failed", "error"):
                err = _attr_or_key(event, "error") or _attr_or_key(event, "response")
                raise RuntimeError(f"codex /responses stream error: {err!r}")
    except RuntimeError:
        raise
    except Exception as exc:
        partial = "".join(text_parts)
        suffix = f"; partial_text={partial[:200]!r}" if partial else ""
        raise RuntimeError(
            f"codex /responses stream interrupted: {exc!r}{suffix}"
        ) from exc

    return _chat_completion_response(
        response_id=response_id or "resp_stream",
        model=response_model,
        content="".join(text_parts),
        finish_reason=finish_reason,
        usage_obj=usage_obj,
    )


def iter_responses_stream_chunks(
    stream: Any, model: str
) -> Iterator[dict[str, Any]]:
    """Yield chat-completion-shaped dict chunks per Responses API SSE event.

    Bridges ChatGPT Codex /responses streaming to the chunk shape AgenticLoop
    expects (chunk["choices"][0]["delta"].{content,reasoning,tool_calls}).
    Used when the caller requests stream=True so tokens reach the UI live
    instead of being drained into a single ModelResponse.
    """
    finish_emitted = False
    call_index_by_id: dict[str, int] = {}
    call_names_by_index: dict[int, str] = {}
    call_arguments_streamed: set[int] = set()

    def _tool_delta(index: int, call_id: str | None = None, name: str = "", arguments: str = "") -> dict[str, Any]:
        return {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }

    def _tool_index(event: Any, item: Any | None = None) -> tuple[int, str | None]:
        raw_id = (
            _attr_or_key(event, "item_id")
            or _attr_or_key(event, "call_id")
            or _attr_or_key(item, "id")
            or _attr_or_key(item, "call_id")
        )
        call_id = str(raw_id) if raw_id else None
        raw_index = _attr_or_key(event, "output_index")
        if raw_index is None:
            raw_index = _attr_or_key(event, "item_index")
        if raw_index is None and call_id in call_index_by_id:
            raw_index = call_index_by_id[call_id]
        if raw_index is None:
            raw_index = len(call_index_by_id)
        index = int(raw_index)
        if call_id:
            call_index_by_id[call_id] = index
        return index, call_id

    try:
        for event in stream:
            event_type = _attr_or_key(event, "type")
            if event_type == "response.output_text.delta":
                delta = _attr_or_key(event, "delta") or ""
                if delta:
                    yield {
                        "choices": [
                            {
                                "delta": {"content": str(delta)},
                                "finish_reason": None,
                            }
                        ]
                    }
            elif event_type == "response.output_item.added":
                # ChatGPT subscription /responses surfaces reasoning as a
                # sealed item — summary array stays empty and only
                # encrypted_content is sent — so reasoning_summary_text.delta
                # never fires. We still want the UI to show "thinking
                # happened", so when the item is reasoning-typed we emit one
                # placeholder onto the reasoning channel.
                item = _attr_or_key(event, "item")
                item_type = _attr_or_key(item, "type") if item is not None else None
                if item_type == "reasoning":
                    yield {
                        "choices": [
                            {
                                "delta": {"reasoning": "[thinking...]"},
                                "finish_reason": None,
                            }
                        ]
                    }
                elif item_type in ("function_call", "tool_call"):
                    index, call_id = _tool_index(event, item)
                    name = str(_attr_or_key(item, "name") or "")
                    call_names_by_index[index] = name
                    yield _tool_delta(index, call_id=call_id, name=name)
            elif event_type in (
                "response.function_call_arguments.delta",
                "response.tool_call_arguments.delta",
                "response.tool_call.arguments.delta",
                "response.output_item.arguments.delta",
            ):
                index, call_id = _tool_index(event)
                delta = str(_attr_or_key(event, "delta") or "")
                if delta:
                    call_arguments_streamed.add(index)
                    yield _tool_delta(
                        index,
                        call_id=call_id,
                        name=call_names_by_index.get(index, ""),
                        arguments=delta,
                    )
            elif event_type in (
                "response.function_call_arguments.done",
                "response.tool_call_arguments.done",
                "response.tool_call.arguments.done",
                "response.output_item.arguments.done",
                "response.output_item.done",
            ):
                item = _attr_or_key(event, "item")
                item_type = _attr_or_key(item, "type") if item is not None else None
                if event_type != "response.output_item.done" or item_type in ("function_call", "tool_call"):
                    index, call_id = _tool_index(event, item)
                    name = str(_attr_or_key(item, "name") or "")
                    arguments = str(
                        _attr_or_key(event, "arguments")
                        or _attr_or_key(item, "arguments")
                        or ""
                    )
                    if name and not call_names_by_index.get(index):
                        call_names_by_index[index] = name
                        yield _tool_delta(index, call_id=call_id, name=name)
                    if (
                        arguments
                        and index not in call_arguments_streamed
                    ):
                        yield _tool_delta(
                            index,
                            call_id=call_id,
                            name=call_names_by_index.get(index, name),
                            arguments=arguments,
                        )
            elif (
                isinstance(event_type, str)
                and event_type.startswith("response.reasoning")
                and event_type.endswith(".delta")
            ):
                # Catch-all so any reasoning-shaped delta (summary_text,
                # reasoning_text, reasoning, etc.) routes through the same
                # reasoning channel AgenticLoop already handles. Some variants
                # use `text` instead of `delta` for the payload field.
                rdelta = (
                    _attr_or_key(event, "delta")
                    or _attr_or_key(event, "text")
                    or ""
                )
                if rdelta:
                    yield {
                        "choices": [
                            {
                                "delta": {"reasoning": str(rdelta)},
                                "finish_reason": None,
                            }
                        ]
                    }
            elif event_type == "response.completed":
                resp_obj = _attr_or_key(event, "response")
                chunk = {
                    "choices": [{"delta": {}, "finish_reason": "stop"}]
                }
                usage = _usage_dict(_attr_or_key(resp_obj, "usage"))
                if usage:
                    chunk["usage"] = usage
                yield chunk
                finish_emitted = True
            elif event_type == "response.incomplete":
                resp_obj = _attr_or_key(event, "response")
                chunk = {
                    "choices": [{"delta": {}, "finish_reason": "length"}]
                }
                usage = _usage_dict(_attr_or_key(resp_obj, "usage"))
                if usage:
                    chunk["usage"] = usage
                yield chunk
                finish_emitted = True
            elif event_type in ("response.failed", "error"):
                err = _attr_or_key(event, "error") or _attr_or_key(
                    event, "response"
                )
                raise RuntimeError(
                    f"codex /responses stream error: {err!r}"
                )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"codex /responses stream interrupted: {exc!r}"
        ) from exc
    if not finish_emitted:
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def build_codex_user_agent(version: str) -> str:
    return jarvis_code_user_agent(version)
