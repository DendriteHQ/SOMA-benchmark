from __future__ import annotations

from typing import Any

from app.miner_tools import available_prompt_ids, compress_text, summarize_text


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

    The proxy now owns prompt mutation. Miners only describe allowed edits:

    - `remove_message`: delete one whole message from the stripped `messages`
      list before forwarding upstream
    - `remove_span`: delete a substring entirely, with no marker inserted
    - `replace_span_with_literal`: replace a substring with one exact allowed
      literal from `/root/SOMA/miner/README_prompting.md`
    - `replace_span_with_generated_text`: replace a substring with text minted
      by a service-owned LLM tool call, referenced by `artifact_id`
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
    - `append_literal`: append one exact allowed template string from
      `/root/SOMA/miner/README_prompting.md`
    - `append_loop_guard`: append one allowed loop-detection reason

    The proxy validates and applies these edits after protected messages are
    stripped, so `message_index` refers to the stripped `messages` list sent to
    this miner.

    You can call service-owned LLM tools from this module:

        available_prompt_ids()
        compress_text(text, prompt_id="compression_faithful")
        summarize_text(text, prompt_id="summary_brief")

    Those tools:
    - use only prompts shipped in `app/available_prompts`
    - return `GeneratedTextResult(artifact_id, text, prompt_id)`
    - automatically count LLM token usage into the `/transform` response

    Example: remove a middle span from the first remaining message.

        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "remove_message",
                    "message_index": 2,
                },
                {
                    "op": "remove_span",
                    "message_index": 0,
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

    Example: replace a span with service-generated compressed text.

        prompt_ids = available_prompt_ids()
        _ = prompt_ids  # inspect prompt inventory if needed
        generated = compress_text(
            "some long span to compress",
            prompt_id="compression_faithful",
        )
        return {
            "contract_version": "soma.cot.edit-plan.v1",
            "edits": [
                {
                    "op": "replace_span_with_generated_text",
                    "message_index": 0,
                    "start": 320,
                    "end": 460,
                    "artifact_id": generated.artifact_id,
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
                    "start": 320,
                    "end": 460,
                    "literal_id": "cmp_source_line",
                    "line_start": 87,
                },
                {
                    "op": "replace_span_with_literal",
                    "message_index": 0,
                    "start": 500,
                    "end": 760,
                    "literal_id": "omitted_source_range",
                    "line_start": 120,
                    "line_end": 148,
                },
            ],
        }
    """
    del payload, messages, path, query, request_id, metadata
    return {
        "contract_version": "soma.cot.edit-plan.v1",
        "edits": [],
    }
