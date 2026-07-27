from __future__ import annotations

from typing import Any

from app.miner_tools import (
    available_prompt_ids,
    compress_text,
    get_candidate_rewrite_artifacts,
    get_rewrite_artifact_by_key,
    get_recent_rewrite_history,
    get_request_rewrite_history,
    summarize_text,
)


def plan_prompt_edits(
    *,
    payload: dict[str, Any] | None = None,
    messages: list[Any] | None = None,
    path: str | None = None,
    query: str | None = None,
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a structured edit plan for prompt/CoT compression.

    The proxy now owns prompt mutation. Miners only describe allowed edits.
    Unless specified otherwise, edits target `message["content"]`. Span/append
    edits may set `target` to `content`, `reasoning`, or `reasoning_details`
    when that field is a string.

    - `remove_message`: delete one whole message from the stripped `messages`
      list before forwarding upstream
    - `remove_message_part`: delete one whole top-level message field such as
      `content`, `reasoning`, or `reasoning_details`
    - `remove_span`: delete a substring entirely, with no marker inserted
    - `replace_span_with_literal`: replace a substring with one exact allowed
      literal from `/root/SOMA/miner/README_prompting.md`
    - `replace_span_with_artifact`: replace a substring with text from a stored
      rewrite artifact, referenced by `artifact_key`
    - `replace_span_with_literal` with `literal_id="cmp_source_line"` and
      `line_start=N`: replace a substring with
      `[[CMP]] source line N [[/CMP]]`
    - `replace_span_with_literal` with `literal_id="omitted_source_range"`
      and `line_start=N`, `line_end=M`: replace a substring with
      `[[Omitted]] source line N ~ source line M Omitted [[/Omitted]]`
    - `wrap_span`: wrap a substring with `[[CMP]]...[[/CMP]]`,
      `[[Omitted]]...[[/Omitted]]`, `[[deleted]]...[[/deleted]]`, or
      `[[BLOCK X]]...[[/BLOCK X]]`
    - `replace_with_block_ref`: replace a substring with
      `Same response as in [[BLOCK X]].`
    - `insert_literal`: insert one exact allowed template string at a specific
      byte position inside a string field
    - `append_literal`: append one exact allowed template string from
      `/root/SOMA/miner/README_prompting.md`
    - `append_loop_guard`: append one allowed loop-detection reason

    The proxy validates and applies these edits after protected messages are
    stripped, so `message_index` refers to the stripped `messages` list sent to
    this miner. Valid targets are `content`, `reasoning`, and
    `reasoning_details`.

    `metadata` includes candidate rewrite artifacts selected by the compression
    service. Each artifact contains:
    - `artifact_key`
    - `prompt_id`
    - `request_url`
    - `model`
    - `source_text`
    - `rewritten_text`
    - `cache_hit`

    Miners should prefer reusing those artifacts when possible instead of
    regenerating rewrites.

    You can call service-owned LLM tools from this module:

        available_prompt_ids()
        compress_text(text, prompt_id="compression_faithful")
        summarize_text(text, prompt_id="summary_brief")
        get_candidate_rewrite_artifacts()
        get_rewrite_artifact_by_key(artifact_key)
        get_request_rewrite_history()
        get_recent_rewrite_history(limit=50, prompt_id=None)

    Those tools:
    - use only prompts shipped in `app/available_prompts`
    - return `GeneratedTextResult(artifact_key, text, prompt_id)`
    - automatically count LLM token usage into the `/transform` response
    - automatically reuse exact cached rewrites for the same
      `(upstream_url, model, prompt_id, original_text)` tuple
    - expose original -> rewritten history so the miner can inspect prior LLM
      edits even when later Copilot/subagent calls omit the previously edited
      span

    Example: remove a middle span from the first remaining message.

        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "remove_message",
                    "message_index": 2,
                },
                {
                    "op": "remove_message_part",
                    "message_index": 1,
                    "target": "reasoning_details",
                },
                {
                    "op": "remove_span",
                    "message_index": 0,
                    "target": "reasoning",
                    "start": 120,
                    "end": 260,
                }
            ],
        }

    Example: define a reusable block, replace a repeated span with a reference,
    and append a loop-detection reason.

        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "wrap_span",
                    "message_index": 1,
                    "start": 0,
                    "end": 180,
                    "template": "block",
                    "block_id": 1,
                },
                {
                    "op": "replace_with_block_ref",
                    "message_index": 2,
                    "start": 40,
                    "end": 220,
                    "block_id": 1,
                },
                {
                    "op": "append_loop_guard",
                    "message_index": 2,
                    "reason": "repeated_tool_call_signature",
                },
            ],
        }

    Example: replace a span with a stored artifact selected for this request.

        candidates = get_candidate_rewrite_artifacts()
        chosen = candidates[0]
        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "replace_span_with_artifact",
                    "message_index": 0,
                    "target": "content",
                    "start": 320,
                    "end": 460,
                    "artifact_key": chosen.artifact_key,
                }
            ],
        }

    Example: create a new artifact and then reference it.

        generated = compress_text(
            "some long span to compress",
            prompt_id="compression_faithful",
        )
        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "replace_span_with_artifact",
                    "message_index": 0,
                    "target": "reasoning",
                    "start": 320,
                    "end": 460,
                    "artifact_key": generated.artifact_key,
                }
            ],
        }

    Example: replace code-like content with a source-line marker.

        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "replace_span_with_literal",
                    "message_index": 0,
                    "target": "content",
                    "start": 320,
                    "end": 460,
                    "literal_id": "cmp_source_line",
                    "line_start": 87,
                },
                {
                    "op": "replace_span_with_literal",
                    "message_index": 0,
                    "target": "content",
                    "start": 500,
                    "end": 760,
                    "literal_id": "omitted_source_range",
                    "line_start": 120,
                    "line_end": 148,
                },
            ],
        }

    Example: insert allowed marker text at a specific position.

        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "insert_literal",
                    "message_index": 0,
                    "target": "content",
                    "position": 320,
                    "literal_id": "cmp_open",
                },
                {
                    "op": "insert_literal",
                    "message_index": 0,
                    "target": "content",
                    "position": 460,
                    "literal_id": "cmp_close",
                },
            ],
        }

    Example: inspect prior rewrite history.

        candidates = get_candidate_rewrite_artifacts()
        by_key = get_rewrite_artifact_by_key(candidates[0].artifact_key) if candidates else None
        request_history = get_request_rewrite_history()
        recent_history = get_recent_rewrite_history(limit=20)
        _ = by_key, request_history, recent_history
    """
    del payload, messages, path, query, request_id, metadata
    return {
        "contract_version": "soma.cot.edit-plan.v1",
        "edits": [],
    }
