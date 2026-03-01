"""Unit tests for RAGPromptBuilder.create_rag_prompt with conversation history."""

import pytest

from src.services.ollama.prompts import RAGPromptBuilder


@pytest.fixture
def builder():
    return RAGPromptBuilder()


@pytest.fixture
def sample_chunks():
    return [
        {"arxiv_id": "2301.00001", "chunk_text": "Transformers use self-attention mechanisms."},
        {"arxiv_id": "2301.00002", "chunk_text": "RAG combines retrieval with generation."},
    ]


@pytest.fixture
def sample_history():
    return [
        {"role": "user", "content": "What is a transformer?"},
        {"role": "assistant", "content": "A transformer is a neural network architecture."},
    ]


# ---------------------------------------------------------------------------
# No history
# ---------------------------------------------------------------------------


def test_no_history_omits_conversation_section(builder, sample_chunks):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks)
    assert "### Conversation History:" not in prompt


def test_none_history_omits_conversation_section(builder, sample_chunks):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks, history=None)
    assert "### Conversation History:" not in prompt


def test_empty_history_list_omits_conversation_section(builder, sample_chunks):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks, history=[])
    assert "### Conversation History:" not in prompt


# ---------------------------------------------------------------------------
# With history
# ---------------------------------------------------------------------------


def test_history_section_present(builder, sample_chunks, sample_history):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks, history=sample_history)
    assert "### Conversation History:" in prompt


def test_history_appears_before_context_section(builder, sample_chunks, sample_history):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks, history=sample_history)
    history_pos = prompt.index("### Conversation History:")
    context_pos = prompt.index("### Context from Papers:")
    assert history_pos < context_pos


def test_history_user_label(builder, sample_chunks, sample_history):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks, history=sample_history)
    assert "User: What is a transformer?" in prompt


def test_history_assistant_label(builder, sample_chunks, sample_history):
    prompt = builder.create_rag_prompt("Tell me more.", sample_chunks, history=sample_history)
    assert "Assistant: A transformer is a neural network architecture." in prompt


def test_history_multiple_turns(builder, sample_chunks):
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    prompt = builder.create_rag_prompt("Q3", sample_chunks, history=history)
    assert "User: Q1" in prompt
    assert "Assistant: A1" in prompt
    assert "User: Q2" in prompt
    assert "Assistant: A2" in prompt


def test_context_and_question_still_present_with_history(builder, sample_chunks, sample_history):
    prompt = builder.create_rag_prompt("Follow-up question?", sample_chunks, history=sample_history)
    assert "### Context from Papers:" in prompt
    assert "### Question:" in prompt
    assert "Follow-up question?" in prompt
    assert "2301.00001" in prompt
    assert "2301.00002" in prompt
