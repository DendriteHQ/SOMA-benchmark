from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "soma.cot.edit-plan.v1"

APPEND_LITERAL_TEMPLATES: dict[str, str] = {
    "compressed_text_starts_here": "Compressed text starts here",
    "compressed_text_ends_here": "Compressed text ends here",
    "cmp_open": "[[CMP]]",
    "cmp_close": "[[/CMP]]",
    "omitted_open": "[[Omitted]]",
    "omitted_close": "[[/Omitted]]",
    "deleted_open": "[[deleted]]",
    "deleted_close": "[[/deleted]]",
}

LOOP_GUARD_TEMPLATES: dict[str, str] = {
    "repeated_assistant_response": "loop_detected: repeated assistant response",
    "repeated_tool_call_signature": "loop_detected: repeated tool call signature",
}

SPAN_MARKER_TEMPLATES: dict[str, tuple[str, str]] = {
    "cmp": ("[[CMP]]", "[[/CMP]]"),
    "omitted": ("[[Omitted]]", "[[/Omitted]]"),
    "deleted": ("[[deleted]]", "[[/deleted]]"),
}


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RemoveSpanEdit(_ContractModel):
    op: Literal["remove_span"]
    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "RemoveSpanEdit":
        if self.end <= self.start:
            raise ValueError("remove_span requires end > start")
        return self


class RemoveMessageEdit(_ContractModel):
    op: Literal["remove_message"]
    message_index: int = Field(ge=0)


class ReplaceSpanWithGeneratedTextEdit(_ContractModel):
    op: Literal["replace_span_with_generated_text"]
    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    artifact_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_span(self) -> "ReplaceSpanWithGeneratedTextEdit":
        if self.end <= self.start:
            raise ValueError("replace_span_with_generated_text requires end > start")
        return self


class ReplaceSpanWithLiteralEdit(_ContractModel):
    op: Literal["replace_span_with_literal"]
    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    literal_id: Literal[
        "compressed_text_starts_here",
        "compressed_text_ends_here",
        "cmp_open",
        "cmp_close",
        "omitted_open",
        "omitted_close",
        "deleted_open",
        "deleted_close",
        "cmp_source_line",
        "omitted_source_range",
    ]
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_span(self) -> "ReplaceSpanWithLiteralEdit":
        if self.end <= self.start:
            raise ValueError("replace_span_with_literal requires end > start")
        if self.literal_id == "cmp_source_line":
            if self.line_start is None:
                raise ValueError("replace_span_with_literal literal_id='cmp_source_line' requires line_start")
            if self.line_end is not None:
                raise ValueError("replace_span_with_literal literal_id='cmp_source_line' does not allow line_end")
        elif self.literal_id == "omitted_source_range":
            if self.line_start is None or self.line_end is None:
                raise ValueError(
                    "replace_span_with_literal literal_id='omitted_source_range' requires line_start and line_end"
                )
            if self.line_end < self.line_start:
                raise ValueError(
                    "replace_span_with_literal literal_id='omitted_source_range' requires line_end >= line_start"
                )
        elif self.line_start is not None or self.line_end is not None:
            raise ValueError("line_start/line_end are only allowed for source-line literal replacements")
        return self


class WrapSpanEdit(_ContractModel):
    op: Literal["wrap_span"]
    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    template: Literal["cmp", "omitted", "deleted", "block"]
    block_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_span(self) -> "WrapSpanEdit":
        if self.end <= self.start:
            raise ValueError("wrap_span requires end > start")
        if self.template == "block" and not self.block_id:
            raise ValueError("wrap_span with template='block' requires block_id")
        if self.template != "block" and self.block_id is not None:
            raise ValueError("block_id is only allowed for wrap_span template='block'")
        return self


class ReplaceWithBlockRefEdit(_ContractModel):
    op: Literal["replace_with_block_ref"]
    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    block_id: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_span(self) -> "ReplaceWithBlockRefEdit":
        if self.end <= self.start:
            raise ValueError("replace_with_block_ref requires end > start")
        return self


class AppendLiteralEdit(_ContractModel):
    op: Literal["append_literal"]
    message_index: int = Field(ge=0)
    literal_id: Literal[
        "compressed_text_starts_here",
        "compressed_text_ends_here",
        "cmp_open",
        "cmp_close",
        "omitted_open",
        "omitted_close",
        "deleted_open",
        "deleted_close",
    ]


class AppendLoopGuardEdit(_ContractModel):
    op: Literal["append_loop_guard"]
    message_index: int = Field(ge=0)
    reason: Literal["repeated_assistant_response", "repeated_tool_call_signature"]


PromptEdit = Annotated[
    RemoveMessageEdit
    | RemoveSpanEdit
    | ReplaceSpanWithGeneratedTextEdit
    | ReplaceSpanWithLiteralEdit
    | WrapSpanEdit
    | ReplaceWithBlockRefEdit
    | AppendLiteralEdit
    | AppendLoopGuardEdit,
    Field(discriminator="op"),
]


class EditPlan(_ContractModel):
    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    edits: list[PromptEdit] = Field(default_factory=list)


class GeneratedTextArtifact(_ContractModel):
    artifact_id: str = Field(min_length=1)
    text: str
    prompt_id: str = Field(min_length=1)


class CompressionUsage(_ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)


class TransformRequestContract(_ContractModel):
    path: str = "/"
    query: str = ""
    request_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TransformResponseContract(_ContractModel):
    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    edits: list[PromptEdit] = Field(default_factory=list)
    generated_texts: list[GeneratedTextArtifact] = Field(default_factory=list)
    compression_usage: CompressionUsage = Field(default_factory=CompressionUsage)


def parse_edit_plan(data: Any) -> EditPlan:
    if isinstance(data, EditPlan):
        return data
    if isinstance(data, TransformResponseContract):
        return EditPlan(contract_version=data.contract_version, edits=list(data.edits))
    if isinstance(data, dict):
        if "edits" not in data:
            raise ValueError("edit plan payload must contain 'edits'")
        return EditPlan.model_validate(data)
    raise TypeError("edit plan must be a dict compatible with EditPlan")


def normalize_edit_plan_output(data: Any) -> TransformResponseContract:
    if isinstance(data, TransformResponseContract):
        return data
    if isinstance(data, dict):
        return TransformResponseContract.model_validate(data)
    plan = parse_edit_plan(data)
    return TransformResponseContract(contract_version=plan.contract_version, edits=list(plan.edits))


def render_append_literal(literal_id: str) -> str:
    if literal_id == "cmp_source_line" or literal_id == "omitted_source_range":
        raise ValueError(f"literal_id {literal_id!r} requires integer line parameters and cannot be appended directly")
    return APPEND_LITERAL_TEMPLATES[literal_id]


def render_loop_guard(reason: str) -> str:
    return LOOP_GUARD_TEMPLATES[reason]


def render_span_wrapper(template: str, block_id: int | None = None) -> tuple[str, str]:
    if template == "block":
        if not block_id:
            raise ValueError("block template requires block_id")
        return (f"[[BLOCK {block_id}]]", f"[[/BLOCK {block_id}]]")
    return SPAN_MARKER_TEMPLATES[template]


def render_literal_replacement(literal_id: str, *, line_start: int | None = None, line_end: int | None = None) -> str:
    if literal_id == "cmp_source_line":
        if line_start is None:
            raise ValueError("cmp_source_line requires line_start")
        return f"[[CMP]] source line {line_start} [[/CMP]]"
    if literal_id == "omitted_source_range":
        if line_start is None or line_end is None:
            raise ValueError("omitted_source_range requires line_start and line_end")
        return f"[[Omitted]] source line {line_start} ~ source line {line_end} Omitted [[/Omitted]]"
    return APPEND_LITERAL_TEMPLATES[literal_id]


def render_block_reference(block_id: int) -> str:
    return f"Same response as in [[BLOCK {block_id}]]."


def _append_text(existing: str, suffix: str) -> str:
    return f"{existing}{suffix}"


def _ensure_message_text(messages: list[Any], message_index: int) -> tuple[dict[str, Any], str]:
    if message_index >= len(messages):
        raise ValueError(f"message_index {message_index} is out of range for payload with {len(messages)} messages")
    message = messages[message_index]
    if not isinstance(message, dict):
        raise ValueError(f"message at index {message_index} is not an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"message at index {message_index} has non-string content and cannot be edited")
    return dict(message), content


def _validate_block_references(edits: list[PromptEdit]) -> None:
    defined_blocks: set[int] = set()
    for edit in edits:
        if isinstance(edit, WrapSpanEdit) and edit.template == "block":
            if edit.block_id in defined_blocks:
                raise ValueError(f"duplicate block_id {edit.block_id!r} in edit plan")
            defined_blocks.add(int(edit.block_id))
    for edit in edits:
        if isinstance(edit, ReplaceWithBlockRefEdit) and edit.block_id not in defined_blocks:
            raise ValueError(f"replace_with_block_ref references unknown block_id {edit.block_id!r}")


def _validate_removed_message_indices(messages: list[Any], edits: list[PromptEdit]) -> set[int]:
    removed_indices: set[int] = set()
    message_count = len(messages)
    for edit in edits:
        if not isinstance(edit, RemoveMessageEdit):
            continue
        if edit.message_index >= message_count:
            raise ValueError(
                f"remove_message has message_index {edit.message_index} out of range for payload with {message_count} messages"
            )
        removed_indices.add(edit.message_index)
    return removed_indices


def _artifact_map(response: TransformResponseContract) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for artifact in response.generated_texts:
        if artifact.artifact_id in artifacts:
            raise ValueError(f"duplicate generated text artifact_id {artifact.artifact_id!r}")
        artifacts[artifact.artifact_id] = artifact.text
    return artifacts


def _validate_generated_text_references(edits: list[PromptEdit], artifacts: dict[str, str]) -> None:
    for edit in edits:
        if isinstance(edit, ReplaceSpanWithGeneratedTextEdit) and edit.artifact_id not in artifacts:
            raise ValueError(f"replace_span_with_generated_text references unknown artifact_id {edit.artifact_id!r}")


def _sorted_span_edits(edits: list[PromptEdit], *, message_length: int, message_index: int) -> list[PromptEdit]:
    span_edits = [
        edit
        for edit in edits
        if isinstance(
            edit,
            (
                RemoveSpanEdit,
                ReplaceSpanWithGeneratedTextEdit,
                ReplaceSpanWithLiteralEdit,
                WrapSpanEdit,
                ReplaceWithBlockRefEdit,
            ),
        )
        and edit.message_index == message_index
    ]
    span_edits.sort(key=lambda item: (item.start, item.end))
    previous_end = -1
    for edit in span_edits:
        if edit.end > message_length:
            raise ValueError(
                f"{edit.op} has end={edit.end} beyond message length {message_length} for message_index {message_index}"
            )
        if edit.start < previous_end:
            raise ValueError(f"overlapping span edits are not allowed for message_index {message_index}")
        previous_end = edit.end
    return span_edits


def apply_edit_plan(payload: dict[str, Any], plan: TransformResponseContract | dict[str, Any]) -> dict[str, Any]:
    response = normalize_edit_plan_output(plan)
    if not response.edits:
        return dict(payload)

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("payload.messages must be a list to apply an edit plan")

    _validate_block_references(response.edits)
    artifacts = _artifact_map(response)
    _validate_generated_text_references(response.edits, artifacts)

    mutated_payload = dict(payload)
    mutated_messages: list[Any] = list(messages)
    removed_message_indices = _validate_removed_message_indices(mutated_messages, response.edits)
    grouped_edits: dict[int, list[PromptEdit]] = {}
    for edit in response.edits:
        if isinstance(edit, RemoveMessageEdit):
            continue
        grouped_edits.setdefault(edit.message_index, []).append(edit)

    for message_index, message_edits in grouped_edits.items():
        if message_index in removed_message_indices:
            raise ValueError(
                f"message_index {message_index} cannot have both remove_message and other edits in the same plan"
            )
        message_copy, content = _ensure_message_text(mutated_messages, message_index)
        span_edits = _sorted_span_edits(message_edits, message_length=len(content), message_index=message_index)

        updated_content = content
        for edit in reversed(span_edits):
            if isinstance(edit, RemoveSpanEdit):
                replacement = ""
            elif isinstance(edit, ReplaceSpanWithGeneratedTextEdit):
                replacement = artifacts[edit.artifact_id]
            elif isinstance(edit, ReplaceSpanWithLiteralEdit):
                replacement = render_literal_replacement(
                    edit.literal_id,
                    line_start=edit.line_start,
                    line_end=edit.line_end,
                )
            elif isinstance(edit, WrapSpanEdit):
                opening, closing = render_span_wrapper(edit.template, block_id=edit.block_id)
                replacement = f"{opening}{updated_content[edit.start:edit.end]}{closing}"
            elif isinstance(edit, ReplaceWithBlockRefEdit):
                replacement = render_block_reference(edit.block_id)
            else:
                continue
            updated_content = f"{updated_content[:edit.start]}{replacement}{updated_content[edit.end:]}"

        append_edits = [
            edit
            for edit in message_edits
            if isinstance(edit, (AppendLiteralEdit, AppendLoopGuardEdit))
        ]
        for edit in append_edits:
            if isinstance(edit, AppendLiteralEdit):
                updated_content = _append_text(updated_content, render_append_literal(edit.literal_id))
            elif isinstance(edit, AppendLoopGuardEdit):
                updated_content = _append_text(updated_content, render_loop_guard(edit.reason))

        message_copy["content"] = updated_content
        mutated_messages[message_index] = message_copy

    mutated_payload["messages"] = [
        message
        for index, message in enumerate(mutated_messages)
        if index not in removed_message_indices
    ]
    return mutated_payload
