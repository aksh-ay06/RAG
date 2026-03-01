"""Integration tests for /stream router session history behaviour.

The streaming endpoint emits Server-Sent Events (SSE) in the form:
  data: <json>\n\n

Event sequence for a normal (non-cached) request:
  1. metadata  — {sources, chunks_used, search_mode, session_id}
  2. chunk(s)  — {chunk: "..."}  (one per Ollama token)
  3. done      — {answer: "...", session_id: "...", done: true}

All external services are replaced with lightweight fakes via
FastAPI dependency_overrides.
"""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.dependencies import (
    get_cache_client,
    get_embeddings_service,
    get_langfuse_tracer,
    get_ollama_client,
    get_opensearch_client,
)
from src.routers.ask import stream_router
from src.schemas.api.ask import AskResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_sse(body: str) -> list[dict]:
    """Parse SSE body into a list of decoded JSON event payloads."""
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_fake_langfuse():
    tracer = MagicMock()

    @contextmanager
    def noop_cm(*args, **kwargs):
        yield None

    tracer.trace_rag_request = noop_cm
    tracer.create_span.return_value = None
    tracer.update_span.return_value = None
    tracer.flush.return_value = None
    return tracer


def make_fake_opensearch(chunks=None):
    if chunks is None:
        chunks = [{"arxiv_id": "2301.00001", "chunk_text": "Attention is all you need."}]
    client = MagicMock()
    client.search_unified.return_value = {"hits": chunks, "total": len(chunks)}
    return client


def make_fake_embeddings():
    svc = AsyncMock()
    svc.embed_query.return_value = [0.1] * 10
    return svc


def make_fake_ollama(tokens=("Hello", " world", ".")):
    """Ollama stub whose generate_rag_answer_stream yields token dicts."""

    async def _stream(*args, **kwargs):
        for tok in tokens:
            yield {"response": tok}
        yield {"done": True}

    ollama = MagicMock()
    ollama.generate_rag_answer_stream = _stream
    return ollama


def make_fake_cache(cached_response=None):
    cache = AsyncMock()
    cache.find_cached_response.return_value = cached_response
    cache.get_session_history.return_value = []
    cache.store_response.return_value = True
    cache.append_to_session_history.return_value = None
    return cache


def make_app(*, opensearch, embeddings, ollama, langfuse_tracer, cache) -> FastAPI:
    app = FastAPI()
    app.include_router(stream_router, prefix="/api/v1")
    app.dependency_overrides[get_opensearch_client] = lambda: opensearch
    app.dependency_overrides[get_embeddings_service] = lambda: embeddings
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    app.dependency_overrides[get_langfuse_tracer] = lambda: langfuse_tracer
    app.dependency_overrides[get_cache_client] = lambda: cache
    return app


@pytest.fixture
def base_payload():
    return {"query": "What is attention?", "top_k": 1, "use_hybrid": False}


async def post_stream(app, payload) -> tuple[int, list[dict]]:
    """POST /stream and return (status_code, list_of_sse_events)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/stream", json=payload)
    return resp.status_code, parse_sse(resp.text)


# ---------------------------------------------------------------------------
# SSE event format
# ---------------------------------------------------------------------------


async def test_stream_emits_metadata_event_first(base_payload):
    """First SSE event must contain sources, chunks_used, and search_mode."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    first = events[0]
    assert "sources" in first
    assert "chunks_used" in first
    assert "search_mode" in first


async def test_stream_metadata_event_includes_session_id(base_payload):
    """Metadata event must include the session_id."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    assert "session_id" in events[0]
    assert events[0]["session_id"] is not None


async def test_stream_emits_chunk_events(base_payload):
    """Token chunks from Ollama must appear as {chunk: ...} events."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("Hello", " world")),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    chunk_events = [e for e in events if "chunk" in e]
    assert len(chunk_events) == 2
    assert chunk_events[0]["chunk"] == "Hello"
    assert chunk_events[1]["chunk"] == " world"


async def test_stream_emits_done_event_last(base_payload):
    """Last SSE event must be the done event with answer and done=True."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("Hi",)),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    last = events[-1]
    assert last.get("done") is True
    assert last.get("answer") == "Hi"


async def test_stream_done_event_includes_session_id(base_payload):
    """Done event must echo the session_id."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    done = events[-1]
    assert "session_id" in done
    assert done["session_id"] == events[0]["session_id"]  # consistent across events


async def test_stream_done_answer_concatenates_tokens(base_payload):
    """The answer in the done event must be all tokens joined."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("Hello", " ", "world")),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    assert events[-1]["answer"] == "Hello world"


async def test_stream_returns_200(base_payload):
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    status, _ = await post_stream(app, base_payload)
    assert status == 200


# ---------------------------------------------------------------------------
# Stateless session behaviour
# ---------------------------------------------------------------------------


async def test_stream_stateless_generates_session_id(base_payload):
    """A stateless request must still get a session_id in the response."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    done = events[-1]
    assert done.get("session_id") is not None


async def test_stream_stateless_does_not_fetch_session_history(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_stream(app, base_payload)

    cache.get_session_history.assert_not_called()


async def test_stream_stateless_checks_exact_match_cache(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_stream(app, base_payload)

    cache.find_cached_response.assert_called_once()


async def test_stream_stateless_stores_response_in_cache(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_stream(app, base_payload)

    cache.store_response.assert_called_once()


async def test_stream_stateless_appends_turn_to_history(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("Generated",)),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_stream(app, base_payload)

    cache.append_to_session_history.assert_called_once()
    _, user_msg, assistant_msg = cache.append_to_session_history.call_args.args
    assert user_msg == base_payload["query"]
    assert assistant_msg == "Generated"


# ---------------------------------------------------------------------------
# Session-aware behaviour
# ---------------------------------------------------------------------------


async def test_stream_session_echoes_session_id(base_payload):
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    payload = {**base_payload, "session_id": "my-stream-session"}
    _, events = await post_stream(app, payload)

    assert events[-1]["session_id"] == "my-stream-session"


async def test_stream_session_fetches_history(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-stream-session"}
    await post_stream(app, payload)

    cache.get_session_history.assert_called_once_with("my-stream-session")


async def test_stream_session_skips_exact_match_cache(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-stream-session"}
    await post_stream(app, payload)

    cache.find_cached_response.assert_not_called()


async def test_stream_session_skips_cache_store(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-stream-session"}
    await post_stream(app, payload)

    cache.store_response.assert_not_called()


async def test_stream_session_appends_turn(base_payload):
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("Session answer",)),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-stream-session"}
    await post_stream(app, payload)

    cache.append_to_session_history.assert_called_once()
    session_id_arg, user_msg, assistant_msg = cache.append_to_session_history.call_args.args
    assert session_id_arg == "my-stream-session"
    assert user_msg == base_payload["query"]
    assert assistant_msg == "Session answer"


# ---------------------------------------------------------------------------
# Cached response path
# ---------------------------------------------------------------------------


async def test_stream_cached_response_served_for_stateless(base_payload):
    """When the cache holds a response, it must be streamed back directly."""
    cached = AskResponse(
        query=base_payload["query"],
        answer="Cached answer text",
        sources=["https://arxiv.org/pdf/2301.00001.pdf"],
        chunks_used=1,
        search_mode="bm25",
    )
    cache = make_fake_cache(cached_response=cached)
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    _, events = await post_stream(app, base_payload)

    done = events[-1]
    assert done.get("cached") is True
    assert done.get("answer") == "Cached answer text"
    assert done.get("done") is True


async def test_stream_cached_response_not_served_for_session(base_payload):
    """Cached path must be skipped when a session_id is present."""
    cached = AskResponse(
        query=base_payload["query"],
        answer="Cached answer text",
        sources=[],
        chunks_used=1,
        search_mode="bm25",
    )
    cache = make_fake_cache(cached_response=cached)
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("Fresh",)),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "active-session"}
    _, events = await post_stream(app, payload)

    done = events[-1]
    assert done.get("cached") is not True
    assert done.get("answer") == "Fresh"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_stream_no_chunks_emits_fallback_done_event(base_payload):
    """When no chunks are found, a single done event with the fallback message is emitted."""
    app = make_app(
        opensearch=make_fake_opensearch(chunks=[]),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    _, events = await post_stream(app, base_payload)

    assert len(events) == 1
    assert events[0].get("done") is True
    assert "No relevant information found" in events[0].get("answer", "")


async def test_stream_cache_failure_does_not_break_request(base_payload):
    """If the cache raises unexpectedly, the stream still completes."""
    cache = make_fake_cache()
    cache.find_cached_response.side_effect = RuntimeError("Redis is down")
    cache.store_response.side_effect = RuntimeError("Redis is down")
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(tokens=("OK",)),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    status, events = await post_stream(app, base_payload)

    assert status == 200
    assert events[-1].get("done") is True
