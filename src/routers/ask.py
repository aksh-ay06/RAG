import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer
from src.services.ollama.prompts import get_prompt_builder

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


async def _load_session_context(
    request: AskRequest,
    session_id: str,
    has_session: bool,
    cache_client,
) -> tuple[List[Dict], Optional[AskResponse]]:
    """Load session history (for session requests) or check the exact-match cache
    (for stateless requests).

    Returns (history, cached_response). Exactly one of the two will be non-empty:
    - history is populated when has_session=True
    - cached_response is set on a stateless cache hit
    """
    if has_session and cache_client:
        try:
            history = await cache_client.get_session_history(session_id)
            return history, None
        except Exception as e:
            logger.warning(f"Failed to fetch session history for {session_id}: {e}")
            return [], None

    if not has_session and cache_client:
        try:
            cached = await cache_client.find_cached_response(request)
            if cached:
                return [], cached
        except Exception as e:
            logger.warning(f"Cache check failed, proceeding normally: {e}")

    return [], None


def _search_mode(use_hybrid: bool) -> str:
    return "hybrid" if use_hybrid else "bm25"


async def _persist_request(
    request: AskRequest,
    response: AskResponse,
    session_id: str,
    has_session: bool,
    cache_client,
) -> None:
    """Persist the completed turn to session history and, for stateless requests,
    to the exact-match cache."""
    if cache_client:
        try:
            await cache_client.append_to_session_history(session_id, request.query, response.answer)
        except Exception as e:
            logger.warning(f"Failed to append to session history for {session_id}: {e}")

    if not has_session and cache_client:
        try:
            await cache_client.store_response(request, response)
        except Exception as e:
            logger.warning(f"Failed to store response in cache: {e}")


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing and exact match caching."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    session_id = request.session_id or uuid.uuid4().hex
    has_session = request.session_id is not None

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            history, cached_response = await _load_session_context(
                request, session_id, has_session, cache_client
            )

            if cached_response:
                logger.info("Returning cached response for exact query match")
                return cached_response.model_copy(update={"cached": True, "session_id": session_id})

            # Retrieve chunks
            chunks, sources, _ = await _prepare_chunks_and_sources(
                request, opensearch_client, embeddings_service, rag_tracer, trace
            )

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in the papers to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode=_search_mode(request.use_hybrid),
                    session_id=session_id,
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt (inject history when present)
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                final_prompt = get_prompt_builder().create_rag_prompt(request.query, chunks, history or None)
                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)

            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode=_search_mode(request.use_hybrid),
                session_id=session_id,
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)
            await _persist_request(request, response, session_id, has_session, cache_client)
            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail="An error occurred processing your request")


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint."""

    session_id = request.session_id or uuid.uuid4().hex
    has_session = request.session_id is not None

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                history, cached_response = await _load_session_context(
                    request, session_id, has_session, cache_client
                )

                if cached_response:
                    logger.info("Returning cached response for exact streaming query match")
                    yield f"data: {json.dumps({'sources': cached_response.sources, 'chunks_used': cached_response.chunks_used, 'search_mode': cached_response.search_mode})}\n\n"
                    text = cached_response.answer
                    for i in range(0, len(text), 20):
                        yield f"data: {json.dumps({'chunk': text[i:i + 20]})}\n\n"
                    yield f"data: {json.dumps({'answer': cached_response.answer, 'cached': True, 'session_id': session_id, 'done': True})}\n\n"
                    return

                # Retrieve chunks
                chunks, sources, _ = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'session_id': session_id, 'done': True})}\n\n"
                    return

                # Send metadata first
                search_mode = _search_mode(request.use_hybrid)
                yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode, 'session_id': session_id})}\n\n"

                # Build prompt (inject history when present)
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    final_prompt = get_prompt_builder().create_rag_prompt(request.query, chunks, history or None)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in ollama_client.generate_rag_answer_stream(
                        query=request.query, chunks=chunks, model=request.model
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'session_id': session_id, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                response = AskResponse(
                    query=request.query,
                    answer=full_response,
                    sources=sources,
                    chunks_used=len(chunks),
                    search_mode=search_mode,
                )
                asyncio.create_task(_persist_request(request, response, session_id, has_session, cache_client))

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'An error occurred processing your request'})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
