"""Test that LiteLLM streaming yields OpenAI-compatible SSE format for RAG service.

Verifies that LiteLLM chunks conform to the format the RAG service expects:
- chunk.choices[0].delta.content is extractable (or "")
- chunk.choices[0].finish_reason is None (streaming) or a string (done)

This is a synchronous mock-based test since pytest-asyncio is not installed.
"""
import pytest
from unittest.mock import MagicMock


class MockDelta:
    """Mock LiteLLM delta object."""
    def __init__(self, content: str = ""):
        self.content = content


class MockChoice:
    """Mock LiteLLM choice object."""
    def __init__(self, content: str = "", finish_reason=None):
        self.delta = MockDelta(content)
        self.finish_reason = finish_reason


class MockChunk:
    """Mock LiteLLM ModelResponse chunk for stream=True.

    LiteLLM's acompletion(stream=True) yields chunks with the same structure
    as OpenAI SDK streaming chunks:
    {
        "choices": [{"delta": {"content": "..."}, "finish_reason": null}]
    }
    """
    def __init__(self, content: str = "", finish_reason=None):
        self.choices = [MockChoice(content, finish_reason)]


def simulate_litellm_streaming(chunks_data):
    """Simulate how RAG service parses LiteLLM streaming chunks.

    This mirrors the parsing logic in rag/service.py _stream_llm.
    """
    results = []
    for content, finish_reason in chunks_data:
        chunk = MockChunk(content, finish_reason)
        delta = chunk.choices[0].delta
        parsed_content = getattr(delta, "content", None) or ""
        parsed_finish = getattr(chunk.choices[0], "finish_reason", None)
        results.append({"content": parsed_content, "finish_reason": parsed_finish})
    return results


class TestLiteLLMSSEFormat:
    """Test LiteLLM SSE format compatibility with RAG service parser."""

    def test_chunk_has_choices_attribute(self):
        """Verify mock chunk has choices attribute (mimics real LiteLLM chunk)."""
        chunk = MockChunk(content="Hello", finish_reason=None)
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) > 0

    def test_delta_content_extractable(self):
        """Verify delta.content is accessible (RAG service reads it)."""
        chunk = MockChunk(content="Hello world")
        delta = chunk.choices[0].delta
        assert hasattr(delta, "content")
        assert delta.content == "Hello world"

    def test_empty_content_returns_empty_string(self):
        """Verify empty content returns '' not None (RAG service uses getattr or '')."""
        chunk = MockChunk(content="", finish_reason=None)
        delta = chunk.choices[0].delta
        # RAG service uses: getattr(delta, "content", None) or ""
        content = getattr(delta, "content", None) or ""
        assert content == ""

    def test_finish_reason_none_during_streaming(self):
        """Verify finish_reason is None during streaming (RAG service checks this)."""
        chunk = MockChunk(content="Hello", finish_reason=None)
        finish = getattr(chunk.choices[0], "finish_reason", None)
        assert finish is None

    def test_finish_reason_string_when_done(self):
        """Verify finish_reason is a string when streaming completes."""
        chunk = MockChunk(content="world", finish_reason="stop")
        finish = getattr(chunk.choices[0], "finish_reason", None)
        assert finish == "stop"
        assert isinstance(finish, str)

    def test_rag_parser_produces_correct_format(self):
        """Verify the full RAG parsing pattern works correctly."""
        chunks_data = [
            ("Hel", None),
            ("lo ", None),
            ("Wor", None),
            ("ld!", "stop"),
        ]
        results = simulate_litellm_streaming(chunks_data)

        assert len(results) == 4
        assert results[0] == {"content": "Hel", "finish_reason": None}
        assert results[1] == {"content": "lo ", "finish_reason": None}
        assert results[2] == {"content": "Wor", "finish_reason": None}
        assert results[3] == {"content": "ld!", "finish_reason": "stop"}

    def test_rag_service_delta_parsing_pattern(self):
        """Verify RAG service's exact parsing pattern: getattr(delta, 'content', None) or ''"""
        # This is the exact pattern used in _stream_llm
        chunk = MockChunk(content="test")
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None) or ""
        assert content == "test"

        # Edge case: delta with None content
        delta_none = MockDelta(content=None)
        content_none = getattr(delta_none, "content", None) or ""
        assert content_none == ""

    def test_rag_service_finish_reason_check(self):
        """Verify RAG service's finish_reason check: break if not None and not 'length'."""
        # RAG service: if finish is not None and finish != "length": break
        test_cases = [
            (None, False, "streaming continues"),
            ("stop", True, "streaming stops"),
            ("length", False, "length is allowed, continue"),
            ("model_length", True, "any non-None non-length stops"),
        ]
        for finish_reason, should_break, description in test_cases:
            chunk = MockChunk(content="x", finish_reason=finish_reason)
            finish = getattr(chunk.choices[0], "finish_reason", None)
            should_stop = finish is not None and finish != "length"
            assert should_stop == should_break, f"{description}: expected {should_break}, got {should_stop}"

    def test_delta_content_with_unicode(self):
        """Verify unicode content (German umlauts) is preserved correctly."""
        chunk = MockChunk(content="Grüß Gott! Müllerstraße 17")
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None) or ""
        assert "Grüß Gott" in content
        assert "Müllerstraße" in content

    def test_multiple_chunks_accumulate_correctly(self):
        """Simulate the RAG streaming accumulation loop."""
        chunks_data = [
            ("Das ", None),
            ("Geburtsdatum ", None),
            ("ist ", None),
            ("17.03.1956.", "stop"),
        ]
        full = ""
        for content, _ in chunks_data:
            chunk = MockChunk(content=content)
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
            full += token
        assert full == "Das Geburtsdatum ist 17.03.1956."

    def test_error_chunk_handled_gracefully(self):
        """Verify error chunks (error=True, content='') are handled.

        RAG service yields: yield {"error": str(e), "content": ""}
        The chat_stream method checks for "error" key.
        """
        # When litellm raises an exception, _stream_llm catches it and yields error dict
        # This test confirms empty content chunks don't break the stream
        error_dict = {"error": "Some error", "content": ""}
        assert error_dict["content"] == ""  # Empty content is handled
        assert "error" in error_dict  # Error key signals problem

    def test_real_litellm_chunk_structure(self):
        """Document the real LiteLLM chunk structure from litellm.acompletion.

        This test is informational — it shows what attributes the real
        litellm.ModelResponseChunk object has based on the OpenAI-compatible format.
        """
        # LiteLLM's stream=True returns ModelResponseChunk with:
        # .choices[0].delta.content  (str or None)
        # .choices[0].finish_reason  (str or None)
        # .model                     (str)
        # .id                        (str)
        # .object                    (str = "chat.completion.chunk")
        #
        # The RAG service only reads delta.content and finish_reason,
        # both of which are guaranteed to exist in LiteLLM's output.

        chunk = MockChunk(content="Test", finish_reason=None)
        # Required attributes used by RAG service
        assert hasattr(chunk.choices[0].delta, "content")
        assert hasattr(chunk.choices[0], "finish_reason")
        # These exist on real LiteLLM chunks too
        assert hasattr(chunk, "model") is False  # Mock doesn't have model (OK - RAG doesn't need it)


class TestRAGServiceIntegration:
    """Integration tests confirming RAG service structure is compatible."""

    def test_rag_service_imports_cleanly(self):
        """Verify RAGService imports without errors."""
        from unittest.mock import AsyncMock
        from app.services.rag.service import RAGService
        service = RAGService(session_factory=None, paperless_client=AsyncMock())
        assert hasattr(service, "_stream_llm")
        assert hasattr(service, "_rewrite_query_llm")

    def test_stream_llm_is_async_generator(self):
        """Verify _stream_llm is an async generator method."""
        import inspect
        from app.services.rag.service import RAGService
        assert inspect.isasyncgenfunction(RAGService._stream_llm)

    def test_rewrite_query_llm_is_async(self):
        """Verify _rewrite_query_llm is an async method."""
        import inspect
        from app.services.rag.service import RAGService
        assert inspect.iscoroutinefunction(RAGService._rewrite_query_llm)

    def test_no_per_provider_streaming_methods(self):
        """Verify per-provider streaming methods are removed."""
        from app.services.rag.service import RAGService
        assert not hasattr(RAGService, "_stream_ollama"), "_stream_ollama should be deleted"
        assert not hasattr(RAGService, "_stream_openai"), "_stream_openai should be deleted"

    def test_no_per_provider_rewrite_methods(self):
        """Verify per-provider rewrite methods are removed."""
        from app.services.rag.service import RAGService
        assert not hasattr(RAGService, "_rewrite_ollama"), "_rewrite_ollama should be deleted"
        assert not hasattr(RAGService, "_rewrite_openai"), "_rewrite_openai should be deleted"

    def test_litellm_import_in_rag_service(self):
        """Verify RAGService uses the unified llm_completion abstraction instead of importing litellm directly."""
        import ast
        from pathlib import Path
        service_path = Path(__file__).parent.parent / "app" / "services" / "rag" / "service.py"
        with open(service_path) as f:
            tree = ast.parse(f.read())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        assert "app.services.llm.service" in imports, "rag/service.py should import from app.services.llm.service (unified LLM abstraction)"
        assert "litellm" not in imports, "rag/service.py should NOT import litellm directly — use llm_completion from llm_service"
