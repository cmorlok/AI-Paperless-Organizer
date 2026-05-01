"""Shared schemas and constants for LiteLLM classifier providers."""

_MAX_TOOL_ROUNDS = 10
_MAX_CONTENT_CHARS = 10000
_LOCAL_LLM_CALL_TIMEOUT = 180.0
_THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "qwq")
_STRICT_SCHEMA_MODELS = ("mistral",)

_SCHEMA_ANALYZE = {
    "type": "object",
    "required": ["title", "correspondent", "created_date", "summary", "language"],
    "properties": {
        "title":         {"type": ["string", "null"]},
        "correspondent": {"type": ["string", "null"]},
        "created_date":  {"type": ["string", "null"]},
        "summary":       {"type": "string"},
        "language":      {"type": "string"},
    },
    "additionalProperties": False,
}

_SCHEMA_TAGS = {
    "type": "object",
    "required": ["tags"],
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_SCHEMA_DOCTYPE = {
    "type": "object",
    "required": ["document_type"],
    "properties": {
        "document_type": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

_SCHEMA_STORAGE_PATH = {
    "type": "object",
    "required": ["path_id", "reason"],
    "properties": {
        "path_id": {"type": ["integer", "null"]},
        "reason":  {"type": "string"},
    },
    "additionalProperties": False,
}

_SCHEMA_VERIFY = {
    "type": "object",
    "properties": {
        "storage_path_id":     {"type": ["integer", "null"]},
        "storage_path_reason": {"type": "string"},
        "tags":                {"type": "array", "items": {"type": "string"}},
        "document_type":       {"type": ["string", "null"]},
        "correspondent":       {"type": ["string", "null"]},
    },
}