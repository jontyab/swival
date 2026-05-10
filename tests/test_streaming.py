"""Tests for REPL streaming preview (call_llm streaming, fmt.streaming_preview)."""

import types
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from swival import fmt
from swival.agent import _completion_with_retry


# -- helpers -----------------------------------------------------------------


def _make_response(content="hello"):
    msg = types.SimpleNamespace(content=content, tool_calls=None, role="assistant")
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


def _make_chunk(delta_content=None, finish_reason=None, *, has_choices=True):
    """Build a fake streaming chunk."""
    if has_choices:
        delta = types.SimpleNamespace(content=delta_content, tool_calls=None)
        choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
        return types.SimpleNamespace(choices=[choice])
    return types.SimpleNamespace(choices=[])


def _chunk_generator(texts, *, include_usage_chunk=False):
    """Yield fake streaming chunks for each text fragment."""
    for t in texts:
        yield _make_chunk(delta_content=t)
    if include_usage_chunk:
        yield _make_chunk(has_choices=False)
    yield _make_chunk(finish_reason="stop")


# -- _completion_with_retry streaming ----------------------------------------


class TestCompletionWithRetryStreaming:
    def test_streaming_calls_callback_per_delta(self):
        chunks = list(_chunk_generator(["Hello", " world"]))
        rebuilt = _make_response("Hello world")

        deltas = []

        with (
            patch("litellm.completion") as mock_comp,
            patch("litellm.stream_chunk_builder", return_value=rebuilt),
        ):
            mock_comp.return_value = iter(chunks)
            response, retries = _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=1,
                verbose=False,
                stream_callback=deltas.append,
            )

        assert deltas == ["Hello", " world"]
        assert response is rebuilt
        assert retries == 0

    def test_streaming_skips_empty_choices_chunks(self):
        chunks = list(_chunk_generator(["Hi"], include_usage_chunk=True))
        rebuilt = _make_response("Hi")
        deltas = []

        with (
            patch("litellm.completion") as mock_comp,
            patch("litellm.stream_chunk_builder", return_value=rebuilt),
        ):
            mock_comp.return_value = iter(chunks)
            response, _ = _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=1,
                verbose=False,
                stream_callback=deltas.append,
            )

        assert deltas == ["Hi"]

    def test_callback_failure_does_not_retry(self):
        chunks = list(_chunk_generator(["A", "B", "C"]))
        rebuilt = _make_response("ABC")

        call_count = 0

        def failing_callback(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("render failed")

        with (
            patch("litellm.completion") as mock_comp,
            patch("litellm.stream_chunk_builder", return_value=rebuilt),
        ):
            mock_comp.return_value = iter(chunks)
            response, retries = _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=3,
                verbose=False,
                stream_callback=failing_callback,
            )

        assert mock_comp.call_count == 1
        assert response is rebuilt
        assert retries == 0

    def test_stream_true_added_to_kwargs(self):
        rebuilt = _make_response("ok")

        with (
            patch("litellm.completion") as mock_comp,
            patch("litellm.stream_chunk_builder", return_value=rebuilt),
        ):
            mock_comp.return_value = iter([_make_chunk(finish_reason="stop")])
            _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=1,
                verbose=False,
                stream_callback=lambda t: None,
            )

        call_kwargs = mock_comp.call_args[1]
        assert call_kwargs["stream"] is True

    def test_original_kwargs_not_mutated(self):
        """stream=True must be added to a copy, not the original dict."""
        rebuilt = _make_response("ok")
        original = {"model": "test", "messages": []}

        with (
            patch("litellm.completion") as mock_comp,
            patch("litellm.stream_chunk_builder", return_value=rebuilt),
        ):
            mock_comp.return_value = iter([_make_chunk(finish_reason="stop")])
            _completion_with_retry(
                original,
                max_retries=1,
                verbose=False,
                stream_callback=lambda t: None,
            )

        assert "stream" not in original

    def test_non_streaming_response_passthrough(self):
        """If provider returns a complete response, handle gracefully."""
        complete = _make_response("direct")

        with patch("litellm.completion", return_value=complete):
            response, retries = _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=1,
                verbose=False,
                stream_callback=lambda t: None,
            )

        assert response is complete

    def test_stream_reset_called_on_retry(self):
        import litellm

        rebuilt = _make_response("ok")
        reset_calls = []

        call_count = 0

        def fake_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise litellm.APIConnectionError(
                    message="dropped",
                    llm_provider="openai",
                    model="test",
                )
            return iter([_make_chunk("ok"), _make_chunk(finish_reason="stop")])

        with (
            patch("litellm.completion", side_effect=fake_completion),
            patch("litellm.stream_chunk_builder", return_value=rebuilt),
        ):
            response, retries = _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=3,
                verbose=False,
                stream_callback=lambda t: None,
                stream_reset=lambda: reset_calls.append(1),
            )

        assert len(reset_calls) == 1
        assert retries == 1

    def test_no_streaming_without_callback(self):
        resp = _make_response("plain")

        with patch("litellm.completion", return_value=resp) as mock_comp:
            response, _ = _completion_with_retry(
                {"model": "test", "messages": []},
                max_retries=1,
                verbose=False,
            )

        call_kwargs = mock_comp.call_args[1]
        assert "stream" not in call_kwargs


# -- call_llm guards --------------------------------------------------------


class TestCallLlmStreamingGuards:
    def test_sanitize_thinking_disables_streaming(self):
        """Streaming must be suppressed when sanitize_thinking is active."""
        from swival.agent import call_llm

        resp = _make_response("clean")
        deltas = []

        with patch("litellm.completion", return_value=resp):
            result = call_llm(
                "http://localhost",
                "openai/test",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                sanitize_thinking=True,
                stream_callback=deltas.append,
                max_retries=1,
            )

        assert deltas == []
        assert result[0].content == "clean"

    def test_secret_shield_disables_streaming(self):
        """Streaming must be suppressed when secret_shield is active."""
        from swival.agent import call_llm

        resp = _make_response("safe")
        deltas = []
        shield = MagicMock()
        shield.encrypt_messages = lambda msgs: msgs
        shield.reverse_known = lambda x: x

        with patch("litellm.completion", return_value=resp):
            call_llm(
                "http://localhost",
                "openai/test",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                secret_shield=shield,
                stream_callback=deltas.append,
                max_retries=1,
            )

        assert deltas == []

    def test_chatgpt_provider_disables_streaming(self):
        """Streaming must be suppressed for the chatgpt provider (litellm#26784)."""
        from swival.agent import call_llm

        resp = _make_response("hi")
        deltas = []

        with (
            patch("litellm.completion", return_value=resp),
            patch("swival.agent._ensure_chatgpt_responses_model_registered"),
        ):
            result = call_llm(
                None,
                "chatgpt/gpt-5.4",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="chatgpt",
                stream_callback=deltas.append,
                max_retries=1,
            )

        assert deltas == []
        assert result[0].content == "hi"


# -- fmt.streaming_preview ---------------------------------------------------


class TestStreamingPreview:
    def test_starts_in_spinner_phase(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview("Thinking (turn 1/5)") as preview:
                assert preview._live is None
        finally:
            fmt._console = old

    def test_transitions_to_live_on_first_text(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview() as preview:
                assert preview._live is None
                preview.update("Hello")
                assert preview._live is not None
        finally:
            fmt._console = old

    def test_empty_update_stays_in_spinner(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview() as preview:
                preview.update("")
                assert preview._live is None
        finally:
            fmt._console = old

    def test_update_accumulates_text(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview() as preview:
                preview.update("Hello")
                preview.update(" world")
                assert "".join(preview._buf) == "Hello world"
        finally:
            fmt._console = old

    def test_reset_clears_buffer_and_returns_to_spinner(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview() as preview:
                preview.update("junk")
                assert preview._live is not None
                preview.reset()
                assert preview._buf == []
                assert preview._live is None
        finally:
            fmt._console = old

    def test_live_is_transient(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(
            file=buf,
            force_terminal=True,
            no_color=True,
            width=80,
        )
        try:
            with fmt.streaming_preview() as preview:
                preview.update("ephemeral")
                assert preview._live.transient is True
        finally:
            fmt._console = old

    def test_exit_from_spinner_phase(self):
        """Exiting without any update (pure spinner) should not error."""
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview("Thinking"):
                pass
        finally:
            fmt._console = old

    def test_label_is_passed_through(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, force_terminal=True, no_color=True, width=80)
        try:
            with fmt.streaming_preview("Thinking (turn 3/10)") as preview:
                assert preview._spinner.suffix == " (turn 3/10)"
        finally:
            fmt._console = old

    def test_stderr_is_terminal_helper(self):
        old = fmt._console
        buf = StringIO()
        fmt._console = Console(file=buf, force_terminal=True, width=80)
        try:
            assert fmt.stderr_is_terminal() is True
        finally:
            fmt._console = old

        fmt._console = Console(file=buf, width=80)
        try:
            assert fmt.stderr_is_terminal() is False
        finally:
            fmt._console = old


# -- cache key isolation -----------------------------------------------------


class TestCacheKeyIsolation:
    def test_cache_hit_skips_streaming(self):
        """A cache hit must not invoke litellm.completion(stream=True)."""
        from swival.agent import call_llm

        cache = MagicMock()
        cache.get.return_value = (
            {"content": "cached", "role": "assistant"},
            "stop",
        )
        deltas = []

        with patch("litellm.completion") as mock_comp:
            result = call_llm(
                "http://localhost",
                "openai/test",
                [{"role": "user", "content": "hi"}],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                cache=cache,
                stream_callback=deltas.append,
                max_retries=1,
            )

        mock_comp.assert_not_called()
        assert deltas == []
        assert result[0].content == "cached"
