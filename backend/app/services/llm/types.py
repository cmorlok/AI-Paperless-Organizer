"""Public LLM response types — stable contract between LLM layer and all callers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string — caller does json.loads()


@dataclass
class LLMResponse:
    content: Optional[str]              # None when finish_reason == "tool_calls"
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    # Pre-cleaned assistant message dict ready for messages.append().
    # Built from model_dump(exclude_none=True) with OpenAI-specific fields
    # (refusal, annotations, audio, function_call) stripped.
    # Only present when tool_calls is not None.
    assistant_message: Optional[dict] = None
