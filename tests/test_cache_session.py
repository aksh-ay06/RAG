"""Unit tests for CacheClient session history methods."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import RedisSettings
from src.services.cache.client import CacheClient


@pytest.fixture
def pipe_mock():
    """Async context manager mock for redis.pipeline().

    Redis pipeline commands (multi, set) are sync; watch/get/execute are awaited.
    """
    pipe = MagicMock()
    pipe.watch = AsyncMock()
    pipe.get = AsyncMock(return_value=None)
    pipe.multi = MagicMock()
    pipe.set = MagicMock()
    pipe.execute = AsyncMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    return pipe


@pytest.fixture
def redis_mock(pipe_mock):
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.pipeline = MagicMock(return_value=pipe_mock)
    return mock


@pytest.fixture
def settings():
    return RedisSettings()


@pytest.fixture
def cache_client(redis_mock, settings):
    return CacheClient(redis_client=redis_mock, settings=settings)


# ---------------------------------------------------------------------------
# get_session_history
# ---------------------------------------------------------------------------


async def test_get_session_history_empty_key(cache_client, redis_mock):
    """Returns [] when the Redis key does not exist."""
    redis_mock.get.return_value = None

    result = await cache_client.get_session_history("sess-abc")

    assert result == []
    redis_mock.get.assert_awaited_once_with("session:sess-abc:history")


async def test_get_session_history_returns_stored_messages(cache_client, redis_mock):
    """Returns the stored list of messages as-is when <= 10 messages."""
    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    ]
    redis_mock.get.return_value = json.dumps(messages)

    result = await cache_client.get_session_history("sess-abc")

    assert result == messages


async def test_get_session_history_trims_to_last_10(cache_client, redis_mock):
    """Caps history at the last 10 messages (5 turns)."""
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    redis_mock.get.return_value = json.dumps(messages)

    result = await cache_client.get_session_history("sess-abc")

    assert len(result) == 10
    assert result == messages[-10:]


async def test_get_session_history_corrupt_json_returns_empty(cache_client, redis_mock):
    """Returns [] when the stored value is not valid JSON."""
    redis_mock.get.return_value = "not valid json {{{"

    result = await cache_client.get_session_history("sess-abc")

    assert result == []


async def test_get_session_history_redis_error_returns_empty(cache_client, redis_mock):
    """Returns [] and logs an error when Redis raises an exception."""
    redis_mock.get.side_effect = ConnectionError("Redis unavailable")

    result = await cache_client.get_session_history("sess-abc")

    assert result == []


# ---------------------------------------------------------------------------
# append_to_session_history
# ---------------------------------------------------------------------------


async def test_append_creates_history_for_new_session(cache_client, pipe_mock):
    """Creates a two-message list when no prior history exists."""
    pipe_mock.get.return_value = None

    await cache_client.append_to_session_history("sess-new", "Hello", "Hi there!")

    pipe_mock.set.assert_called_once()
    key, payload = pipe_mock.set.call_args.args
    assert key == "session:sess-new:history"
    stored = json.loads(payload)
    assert stored == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


async def test_append_accumulates_turns(cache_client, pipe_mock):
    """Appends a new turn to an existing history."""
    existing = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    pipe_mock.get.return_value = json.dumps(existing)

    await cache_client.append_to_session_history("sess-abc", "Second question", "Second answer")

    _, payload = pipe_mock.set.call_args.args
    stored = json.loads(payload)
    assert len(stored) == 4
    assert stored[2] == {"role": "user", "content": "Second question"}
    assert stored[3] == {"role": "assistant", "content": "Second answer"}


async def test_append_sets_correct_ttl(cache_client, pipe_mock, settings):
    """Sets the Redis key with the session_ttl_hours TTL."""
    pipe_mock.get.return_value = None
    expected_ttl = int(timedelta(hours=settings.session_ttl_hours).total_seconds())

    await cache_client.append_to_session_history("sess-ttl", "q", "a")

    kwargs = pipe_mock.set.call_args.kwargs
    assert kwargs["ex"] == expected_ttl


async def test_append_redis_error_does_not_raise(cache_client, pipe_mock):
    """Swallows Redis errors rather than propagating them."""
    pipe_mock.get.side_effect = ConnectionError("Redis unavailable")

    # Should not raise
    await cache_client.append_to_session_history("sess-err", "q", "a")
