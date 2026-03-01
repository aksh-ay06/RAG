"""Integration tests for /ask router session history behaviour.

All external services are replaced with lightweight fakes via
FastAPI dependency_overrides, so no real Redis / OpenSearch / Ollama
is required.
"""

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
from src.routers.ask import ask_router


# ---------------------------------------------------------------------------
# Minimal app fixture
# ---------------------------------------------------------------------------


def make_app(
    *,
    opensearch,
    embeddings,
    ollama,
    langfuse_tracer,
    cache,
) -> FastAPI:
    """Build a stripped-down FastAPI app with dependency overrides."""
    app = FastAPI()
    app.include_router(ask_router, prefix="/api/v1")

    app.dependency_overrides[get_opensearch_client] = lambda: opensearch
    app.dependency_overrides[get_embeddings_service] = lambda: embeddings
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    app.dependency_overrides[get_langfuse_tracer] = lambda: langfuse_tracer
    app.dependency_overrides[get_cache_client] = lambda: cache

    return app


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


def make_fake_langfuse():
    """LangfuseTracer with no real Langfuse client (all ops are no-ops)."""
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
    """OpenSearch client stub returning the given chunks."""
    if chunks is None:
        chunks = [{"arxiv_id": "2301.00001", "chunk_text": "Attention is all you need."}]
    client = MagicMock()
    client.search_unified.return_value = {
        "hits": chunks,
        "total": len(chunks),
    }
    return client


def make_fake_embeddings():
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.1] * 10
    return embeddings


def make_fake_ollama(answer="This is the generated answer."):
    ollama = AsyncMock()
    ollama.generate_rag_answer.return_value = {"answer": answer}
    return ollama


def make_fake_cache():
    cache = AsyncMock()
    cache.find_cached_response.return_value = None
    cache.get_session_history.return_value = []
    cache.store_response.return_value = True
    cache.append_to_session_history.return_value = None
    return cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def base_payload():
    return {"query": "What is attention?", "top_k": 1, "use_hybrid": False}


async def post_ask(app, payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/api/v1/ask", json=payload)


# ---------------------------------------------------------------------------
# Tests: stateless (no session_id in request)
# ---------------------------------------------------------------------------


async def test_stateless_response_includes_session_id(base_payload):
    """Every response must include a session_id even when none was sent."""
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    resp = await post_ask(app, base_payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["session_id"] is not None


async def test_stateless_request_does_not_fetch_session_history(base_payload):
    """When no session_id is sent, get_session_history must never be called."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_ask(app, base_payload)

    cache.get_session_history.assert_not_called()


async def test_stateless_request_checks_exact_match_cache(base_payload):
    """Stateless requests should hit the exact-match cache."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_ask(app, base_payload)

    cache.find_cached_response.assert_called_once()


async def test_stateless_request_stores_response_in_cache(base_payload):
    """Stateless responses should be written to the exact-match cache."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_ask(app, base_payload)

    cache.store_response.assert_called_once()


async def test_stateless_request_appends_turn_to_history(base_payload):
    """Even stateless turns must be saved so the client can continue the session."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama("Generated answer"),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    await post_ask(app, base_payload)

    cache.append_to_session_history.assert_called_once()
    _, user_msg, assistant_msg = cache.append_to_session_history.call_args.args
    assert user_msg == base_payload["query"]
    assert assistant_msg == "Generated answer"


# ---------------------------------------------------------------------------
# Tests: session-aware (session_id provided)
# ---------------------------------------------------------------------------


async def test_session_request_returns_same_session_id(base_payload):
    """The response must echo back the same session_id that was sent."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-session-123"}
    resp = await post_ask(app, payload)

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-session-123"


async def test_session_request_fetches_history(base_payload):
    """When session_id is provided, get_session_history must be called with it."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-session-123"}
    await post_ask(app, payload)

    cache.get_session_history.assert_called_once_with("my-session-123")


async def test_session_request_skips_exact_match_cache(base_payload):
    """Session requests must not consult the exact-match cache."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-session-123"}
    await post_ask(app, payload)

    cache.find_cached_response.assert_not_called()


async def test_session_request_skips_cache_store(base_payload):
    """Session responses must not be written to the exact-match cache."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-session-123"}
    await post_ask(app, payload)

    cache.store_response.assert_not_called()


async def test_session_request_appends_turn_to_history(base_payload):
    """Each session turn must be appended with the correct session_id."""
    cache = make_fake_cache()
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama("Session answer"),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    payload = {**base_payload, "session_id": "my-session-123"}
    await post_ask(app, payload)

    cache.append_to_session_history.assert_called_once()
    session_id_arg, user_msg, assistant_msg = cache.append_to_session_history.call_args.args
    assert session_id_arg == "my-session-123"
    assert user_msg == base_payload["query"]
    assert assistant_msg == "Session answer"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


async def test_no_chunks_returns_fallback_answer(base_payload):
    """When OpenSearch returns nothing, the fallback message is returned."""
    opensearch = make_fake_opensearch(chunks=[])
    app = make_app(
        opensearch=opensearch,
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=make_fake_cache(),
    )
    resp = await post_ask(app, base_payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_used"] == 0
    assert "session_id" in data


async def test_cache_failure_does_not_break_request(base_payload):
    """If the cache raises unexpectedly, the request still succeeds."""
    cache = make_fake_cache()
    cache.find_cached_response.side_effect = RuntimeError("Redis is down")
    cache.store_response.side_effect = RuntimeError("Redis is down")
    app = make_app(
        opensearch=make_fake_opensearch(),
        embeddings=make_fake_embeddings(),
        ollama=make_fake_ollama(),
        langfuse_tracer=make_fake_langfuse(),
        cache=cache,
    )
    resp = await post_ask(app, base_payload)

    assert resp.status_code == 200
