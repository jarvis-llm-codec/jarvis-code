import argparse
import json
from pathlib import Path

import pytest

import longhaul_runner as runner


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_schema_records(schema_name: str, records: list[dict]) -> None:
    schema_path = Path("C:/longhaul-bench/spec/schemas") / schema_name
    if not schema_path.exists():
        pytest.skip(f"missing local longhaul schema: {schema_path}")
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for record in records:
        validator.validate(record)


def test_prepare_writes_jsonl_and_preserves_multiline(tmp_path: Path) -> None:
    script_path = tmp_path / "script.jsonl"
    first_prompt = "alpha\nbeta\n  gamma"
    write_jsonl(
        script_path,
        [
            {"turn": 1, "kind": "filler", "text": first_prompt},
            {"turn": 2, "kind": "probe", "text": "tail prompt"},
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"tier": 100, "total_turns": 2, "workload_turns": 1}),
        encoding="utf-8",
    )
    workdir = tmp_path / "work"

    rc = runner.prepare_command(argparse.Namespace(script=str(script_path), out=str(workdir), seed=42, tier=None))

    assert rc == 0
    prompt_file = workdir / "prompts" / "full" / "auto_prompts.jsonl"
    prompts = read_jsonl(prompt_file)
    assert prompts[0]["text"] == first_prompt
    assert prompts[0]["delivered_sha256"] == runner.sha256_text(first_prompt)

    prompt_map = read_jsonl(workdir / "prompt_map.jsonl")
    assert prompt_map[0]["delivered_sha256"] == runner.sha256_text(first_prompt)
    assert prompt_map[0]["flattened"] is False
    assert prompt_map[0]["has_newlines"] is True

    meta = json.loads((workdir / "longhaul_runner_meta.json").read_text(encoding="utf-8"))
    assert meta["auto_prompt_mode"] == "jsonl_text"
    assert meta["bench_conv"] == "longhaul-s42-t100"
    assert meta["prompt_flattening"] is False
    assert Path(meta["manifest_copy_path"]).exists()

    _label, limited_file, target = runner.active_prompt_file_for_run(workdir, prompt_map, 1)
    assert target == 1
    assert limited_file.name == "auto_prompts.jsonl"
    assert read_jsonl(limited_file)[0]["text"] == first_prompt


def test_normalize_provider_usage_snapshots_handles_cache_reasoning_and_cumulative() -> None:
    per_turn = runner.provider_usage_snapshot(
        {
            "llm_meta": {
                "provider": "openai-codex",
                "api": "openai-codex-responses",
                "provider_calls": 2,
                "provider_usage_scope": "pi_turn_summed_per_turn",
                "provider_call_usages": [
                    {"input": 60, "output": 12, "cacheRead": 10, "cacheWrite": 0, "reasoningTokens": 3},
                    {"input": 40, "output": 8, "cacheRead": 20, "cacheWrite": 5, "reasoningTokens": 4},
                ],
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 30,
                    "cacheWrite": 5,
                    "reasoningTokens": 7,
                    "totalTokens": 162,
                },
            }
        }
    )
    normalized = runner.normalize_provider_usage_snapshots([per_turn])[0]
    assert normalized["input_tokens"] == 135
    assert normalized["output_tokens"] == 27
    assert normalized["provider_context_tokens"] == 70
    assert normalized["provider_calls"] == 2

    missing_call_breakdown = dict(per_turn)
    missing_call_breakdown["provider_call_usages"] = []
    fallback = runner.normalize_provider_usage_snapshots([missing_call_breakdown])[0]
    assert fallback["input_tokens"] == 135
    assert fallback["provider_context_tokens"] is None
    assert fallback["provider_context_tokens_unavailable_reason"] == "missing_provider_call_usages_for_multi_call_turn"

    cumulative = runner.normalize_provider_usage_snapshots(
        [
            {
                "scope": "provider_cumulative",
                "source": "test",
                "input_uncached_tokens": 100,
                "cache_read_tokens": 10,
                "cache_write_tokens": 0,
                "output_visible_tokens": 20,
                "reasoning_tokens": 5,
                "total_tokens": 135,
            },
            {
                "scope": "provider_cumulative",
                "source": "test",
                "input_uncached_tokens": 250,
                "cache_read_tokens": 40,
                "cache_write_tokens": 0,
                "output_visible_tokens": 55,
                "reasoning_tokens": 15,
                "total_tokens": 360,
            },
        ]
    )
    assert cumulative[0]["input_tokens"] == 110
    assert cumulative[0]["output_tokens"] == 25
    assert cumulative[1]["input_tokens"] == 180
    assert cumulative[1]["output_tokens"] == 45


def test_collect_writes_v04_ledger_from_raw_usage(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    out_dir = tmp_path / "out"
    store = tmp_path / "bench_archive"
    bench_conv = "longhaul-s42-t100"
    prompts = ["first\nprompt", "second prompt"]
    responses = ["answer one", "answer two"]

    write_jsonl(
        workdir / "prompts" / "full" / "auto_prompts.jsonl",
        [
            {"turn": 1, "kind": "filler", "text": prompts[0], "delivered_sha256": runner.sha256_text(prompts[0])},
            {"turn": 2, "kind": "probe", "text": prompts[1], "delivered_sha256": runner.sha256_text(prompts[1])},
        ],
    )
    write_jsonl(
        workdir / "prompt_map.jsonl",
        [
            {
                "turn": 1,
                "kind": "filler",
                "delivered_sha256": runner.sha256_text(prompts[0]),
                "prompt_sha256": runner.sha256_text(prompts[0]),
                "original_sha256": runner.sha256_text(prompts[0]),
                "flattened": False,
            },
            {
                "turn": 2,
                "kind": "probe",
                "delivered_sha256": runner.sha256_text(prompts[1]),
                "prompt_sha256": runner.sha256_text(prompts[1]),
                "original_sha256": runner.sha256_text(prompts[1]),
                "flattened": False,
            },
        ],
    )
    runner.write_json(
        workdir / "longhaul_runner_meta.json",
        {
            "auto_prompt_mode": "jsonl_text",
            "bench_conv": bench_conv,
            "prompt_count": 2,
            "seed": 42,
            "tier": 100,
        },
    )
    runner.write_json(
        workdir / "longhaul_runner_run_state.json",
        {
            "active_label": "full",
            "active_prompts_path": str(workdir / "prompts" / "full" / "auto_prompts.jsonl"),
            "started_at": "2026-01-01T00:00:00+00:00",
            "target_turns": 2,
        },
    )
    write_jsonl(
        store / f"{bench_conv}.jsonl",
        [
            {
                "timestamp": "2026-01-01T00:00:01+00:00",
                "user": prompts[0],
                "assistant": responses[0],
                "llm_meta": {
                    "provider": "openai-codex",
                    "api": "openai-codex-responses",
                    "response_model": "gpt-5.5",
                    "provider_calls": 2,
                    "provider_usage_scope": "pi_turn_summed_per_turn",
                    "provider_call_usages": [
                        {"input": 6, "output": 2, "cacheRead": 2, "cacheWrite": 1, "reasoningTokens": 1},
                        {"input": 4, "output": 2, "cacheRead": 0, "cacheWrite": 0, "reasoningTokens": 2},
                    ],
                    "usage": {
                        "input": 10,
                        "output": 4,
                        "cacheRead": 2,
                        "cacheWrite": 1,
                        "reasoningTokens": 3,
                        "totalTokens": 20,
                    },
                },
            },
            {
                "timestamp": "2026-01-01T00:00:01+00:00",
                "turn": 1,
                "kind": "encoder",
                "encoder_meta": {"enc_in": 111, "enc_out": 22, "enc_think": 3, "failure_mode": "ok", "encoder_retries": 0},
            },
            {
                "timestamp": "2026-01-01T00:00:02+00:00",
                "user": prompts[1],
                "assistant": responses[1],
                "llm_meta": {
                    "provider": "openai-codex",
                    "api": "openai-codex-responses",
                    "response_model": "gpt-5.5",
                    "provider_calls": 1,
                    "provider_usage_scope": "pi_turn_summed_per_turn",
                    "usage": {"input": 30, "output": 8, "cacheRead": 5, "cacheWrite": 0, "totalTokens": 43},
                },
            },
        ],
    )

    rc = runner.collect_command(
        argparse.Namespace(
            workdir=str(workdir),
            out=str(out_dir),
            raw_bench_store=str(store),
            model_label="gpt-5.5 test",
        )
    )

    assert rc == 0
    ledger = read_jsonl(out_dir / "ledger.jsonl")
    assert ledger[0]["input_tokens"]["provider"] == 13
    assert ledger[0]["output_tokens"]["provider"] == 7
    assert ledger[0]["provider_context_tokens"] == 9
    assert ledger[0]["expected_input_tokens"] == runner.estimate_tokens(prompts[0])
    assert ledger[1]["input_tokens"]["provider"] == 35
    assert ledger[1]["expected_input_tokens"] == (
        runner.estimate_tokens(prompts[0]) + runner.estimate_tokens(responses[0]) + runner.estimate_tokens(prompts[1])
    )
    assert "provider_source=raw_bench_archive.llm_meta.usage" in ledger[0]["notes"]
    assert "provider_context=max_call(input+cacheRead+cacheWrite)" in ledger[0]["notes"]
    assert "provider_total_check=matches_totalTokens" in ledger[0]["notes"]
    assert "provider_scope=chat_main_loop_only" in ledger[0]["notes"]
    assert "encoder_aux_excluded_from_provider" in ledger[0]["notes"]
    assert "encoder_estimate_in=111" in ledger[0]["notes"]
    assert "encoder_estimate_out=22" in ledger[0]["notes"]

    transcript = read_jsonl(out_dir / "transcript.jsonl")
    assert transcript[0]["prompt_sha256"] == runner.sha256_text(prompts[0])

    run_meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["provider_token_source"] == runner.PROVIDER_TOKEN_SOURCE
    assert "encoder_aux_excluded_from_provider" in run_meta["token_ledger_policy"]
    assert run_meta["dnf"] is False

    validate_schema_records("ledger.schema.json", ledger)
    validate_schema_records("transcript.schema.json", transcript)
    validate_schema_records("run_meta.schema.json", [run_meta])
