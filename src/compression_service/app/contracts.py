from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "soma.cot.edit-plan.v1"
EDIT_STRING_TARGET_FIELDS = ("content", "reasoning", "reasoning_details")
EditTargetField = Literal["content", "reasoning", "reasoning_details"]

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
    target: EditTargetField = "content"
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


class RemoveMessagePartEdit(_ContractModel):
    op: Literal["remove_message_part"]
    message_index: int = Field(ge=0)
    target: EditTargetField


class ReplaceSpanWithArtifactEdit(_ContractModel):
    op: Literal["replace_span_with_artifact"]
    message_index: int = Field(ge=0)
    target: EditTargetField = "content"
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    artifact_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_span(self) -> "ReplaceSpanWithArtifactEdit":
        if self.end <= self.start:
            raise ValueError("replace_span_with_artifact requires end > start")
        return self


class ReplaceSpanWithLiteralEdit(_ContractModel):
    op: Literal["replace_span_with_literal"]
    message_index: int = Field(ge=0)
    target: EditTargetField = "content"
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
    target: EditTargetField = "content"
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
    target: EditTargetField = "content"
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
    target: EditTargetField = "content"
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


class InsertLiteralEdit(_ContractModel):
    op: Literal["insert_literal"]
    message_index: int = Field(ge=0)
    target: EditTargetField = "content"
    position: int = Field(ge=0)
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
    target: EditTargetField = "content"
    reason: Literal["repeated_assistant_response", "repeated_tool_call_signature"]


PromptEdit = Annotated[
    RemoveMessageEdit
    | RemoveMessagePartEdit
    | RemoveSpanEdit
    | ReplaceSpanWithArtifactEdit
    | ReplaceSpanWithLiteralEdit
    | WrapSpanEdit
    | ReplaceWithBlockRefEdit
    | AppendLiteralEdit
    | InsertLiteralEdit
    | AppendLoopGuardEdit,
    Field(discriminator="op"),
]


class EditPlan(_ContractModel):
    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    edits: list[PromptEdit] = Field(default_factory=list)


class RewriteArtifactContract(_ContractModel):
    artifact_key: str = Field(min_length=1)
    source_text: str
    text: str
    prompt_id: str = Field(min_length=1)
    request_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    cache_hit: bool = False


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
    artifacts: list[RewriteArtifactContract] = Field(default_factory=list)
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


def _ensure_message_text(
    messages: list[Any],
    message_index: int,
    *,
    target: EditTargetField,
) -> tuple[dict[str, Any], str]:
    if message_index >= len(messages):
        raise ValueError(f"message_index {message_index} is out of range for payload with {len(messages)} messages")
    message = messages[message_index]
    if not isinstance(message, dict):
        raise ValueError(f"message at index {message_index} is not an object")
    value = message.get(target)
    if not isinstance(value, str):
        raise ValueError(f"message at index {message_index} has non-string {target!r} and cannot be edited")
    return dict(message), value


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


def _validate_removed_message_part_targets(messages: list[Any], edits: list[PromptEdit]) -> dict[int, set[EditTargetField]]:
    removed_targets_by_message: dict[int, set[EditTargetField]] = {}
    message_count = len(messages)
    for edit in edits:
        if not isinstance(edit, RemoveMessagePartEdit):
            continue
        if edit.message_index >= message_count:
            raise ValueError(
                f"remove_message_part has message_index {edit.message_index} out of range for payload with {message_count} messages"
            )
        message = messages[edit.message_index]
        if not isinstance(message, dict):
            raise ValueError(f"message at index {edit.message_index} is not an object")
        if edit.target not in message:
            raise ValueError(
                f"remove_message_part target {edit.target!r} is missing on message_index {edit.message_index}"
            )
        removed_targets_by_message.setdefault(edit.message_index, set()).add(edit.target)
    return removed_targets_by_message


def _artifact_map(response: TransformResponseContract) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for artifact in response.artifacts:
        if artifact.artifact_key in artifacts:
            raise ValueError(f"duplicate artifact_key {artifact.artifact_key!r}")
        artifacts[artifact.artifact_key] = artifact.text
    return artifacts


def _validate_artifact_references(edits: list[PromptEdit], artifacts: dict[str, str]) -> None:
    for edit in edits:
        if isinstance(edit, ReplaceSpanWithArtifactEdit) and edit.artifact_key not in artifacts:
            raise ValueError(f"replace_span_with_artifact references unknown artifact_key {edit.artifact_key!r}")


def _sorted_positioned_edits(
    edits: list[PromptEdit],
    *,
    message_length: int,
    message_index: int,
    target: EditTargetField,
) -> list[PromptEdit]:
    positioned_edits = [
        edit
        for edit in edits
        if isinstance(
            edit,
            (
                RemoveSpanEdit,
                ReplaceSpanWithArtifactEdit,
                ReplaceSpanWithLiteralEdit,
                WrapSpanEdit,
                ReplaceWithBlockRefEdit,
                InsertLiteralEdit,
            ),
        )
        and edit.message_index == message_index
        and edit.target == target
    ]
    positioned_edits.sort(
        key=lambda item: (
            item.position if isinstance(item, InsertLiteralEdit) else item.start,
            item.position if isinstance(item, InsertLiteralEdit) else item.end,
        )
    )
    previous_end = -1
    for edit in positioned_edits:
        if isinstance(edit, InsertLiteralEdit):
            if edit.position > message_length:
                raise ValueError(
                    f"insert_literal has position={edit.position} beyond message length {message_length} for message_index {message_index}"
                )
            if edit.position < previous_end:
                raise ValueError(f"insert_literal cannot be placed inside another edited span for message_index {message_index}")
            continue
        if edit.end > message_length:
            raise ValueError(
                f"{edit.op} has end={edit.end} beyond message length {message_length} for message_index {message_index}"
            )
        if edit.start < previous_end:
            raise ValueError(f"overlapping span edits are not allowed for message_index {message_index}")
        previous_end = edit.end
    return positioned_edits


def apply_edit_plan(payload: dict[str, Any], plan: TransformResponseContract | dict[str, Any]) -> dict[str, Any]:
    response = normalize_edit_plan_output(plan)
    if not response.edits:
        return dict(payload)

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("payload.messages must be a list to apply an edit plan")

    _validate_block_references(response.edits)
    artifacts = _artifact_map(response)
    _validate_artifact_references(response.edits, artifacts)

    mutated_payload = dict(payload)
    mutated_messages: list[Any] = list(messages)
    removed_message_indices = _validate_removed_message_indices(mutated_messages, response.edits)
    removed_message_part_targets = _validate_removed_message_part_targets(mutated_messages, response.edits)
    grouped_edits: dict[int, list[PromptEdit]] = {}
    for edit in response.edits:
        if isinstance(edit, (RemoveMessageEdit, RemoveMessagePartEdit)):
            continue
        grouped_edits.setdefault(edit.message_index, []).append(edit)

    for message_index, message_edits in grouped_edits.items():
        if message_index in removed_message_indices:
            raise ValueError(
                f"message_index {message_index} cannot have both remove_message and other edits in the same plan"
            )
        message_copy = dict(mutated_messages[message_index]) if isinstance(mutated_messages[message_index], dict) else {}
        removed_targets = removed_message_part_targets.get(message_index, set())
        for target in removed_targets:
            message_copy.pop(target, None)
        for target in EDIT_STRING_TARGET_FIELDS:
            target_edits = [edit for edit in message_edits if getattr(edit, "target", "content") == target]
            if not target_edits:
                continue
            if target in removed_targets:
                raise ValueError(
                    f"message_index {message_index} target {target!r} cannot have both remove_message_part and other edits"
                )
            message_copy, content = _ensure_message_text(mutated_messages, message_index, target=target)
            positioned_edits = _sorted_positioned_edits(
                target_edits,
                message_length=len(content),
                message_index=message_index,
                target=target,
            )

            updated_content = content
            for edit in reversed(positioned_edits):
                if isinstance(edit, InsertLiteralEdit):
                    replacement = render_append_literal(edit.literal_id)
                    updated_content = f"{updated_content[:edit.position]}{replacement}{updated_content[edit.position:]}"
                    continue
                if isinstance(edit, RemoveSpanEdit):
                    replacement = ""
                elif isinstance(edit, ReplaceSpanWithArtifactEdit):
                    replacement = artifacts[edit.artifact_key]
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
                for edit in target_edits
                if isinstance(edit, (AppendLiteralEdit, AppendLoopGuardEdit))
            ]
            for edit in append_edits:
                if isinstance(edit, AppendLiteralEdit):
                    updated_content = _append_text(updated_content, render_append_literal(edit.literal_id))
                elif isinstance(edit, AppendLoopGuardEdit):
                    updated_content = _append_text(updated_content, render_loop_guard(edit.reason))

            message_copy[target] = updated_content
        mutated_messages[message_index] = message_copy

    mutated_payload["messages"] = [
        message
        for index, message in enumerate(mutated_messages)
        if index not in removed_message_indices
    ]
    return mutated_payload
