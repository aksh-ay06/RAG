import json
import logging
from typing import AsyncIterator, Optional

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"


async def _stream_chat_turn(
    message: str,
    history: list,
    session_id: Optional[str],
    top_k: int,
    use_hybrid: bool,
    model: str,
    categories: str,
) -> AsyncIterator[tuple[list, Optional[str]]]:
    """Yield (updated_history, session_id) for each streaming chunk."""
    category_list = [c.strip() for c in categories.split(",") if c.strip()] if categories else None

    payload: dict = {
        "query": message,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "categories": category_list,
    }
    if session_id:
        payload["session_id"] = session_id

    # Append user turn; assistant slot starts empty
    history = history + [[message, ""]]
    current_answer = ""
    sources: list[str] = []
    chunks_used = 0
    search_mode = ""
    new_session_id = session_id

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/stream",
                json=payload,
                headers={"Accept": "text/plain"},
            ) as response:
                if response.status_code != 200:
                    history[-1][1] = f"Error: API returned status {response.status_code}"
                    yield history, new_session_id
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        history[-1][1] = f"Error: {data['error']}"
                        yield history, new_session_id
                        return

                    # Metadata event (sources, session_id)
                    if "sources" in data:
                        sources = data["sources"]
                        chunks_used = data.get("chunks_used", 0)
                        search_mode = data.get("search_mode", "unknown")
                        new_session_id = data.get("session_id", new_session_id)
                        continue

                    # Streaming text chunk
                    if "chunk" in data:
                        current_answer += data["chunk"]
                        history[-1][1] = current_answer
                        yield history, new_session_id

                    # Done event
                    if data.get("done"):
                        new_session_id = data.get("session_id", new_session_id)
                        final = data.get("answer", current_answer)
                        if final != current_answer:
                            current_answer = final

                        # Append source footer
                        formatted = current_answer
                        if sources or chunks_used:
                            formatted += (
                                f"\n\n---\n*{search_mode} · {chunks_used} chunk"
                                f"{'s' if chunks_used != 1 else ''}*"
                            )
                            if sources:
                                formatted += f" · {len(sources)} source{'s' if len(sources) != 1 else ''}\n"
                                for i, src in enumerate(sources[:3], 1):
                                    label = src.split("/")[-1]
                                    formatted += f"\n{i}. [{label}]({src})"
                                if len(sources) > 3:
                                    formatted += f"\n… and {len(sources) - 3} more"

                        history[-1][1] = formatted
                        yield history, new_session_id
                        break

    except httpx.RequestError as e:
        history[-1][1] = (
            f"Connection error: {e}\n\nMake sure the API server is running at `{API_BASE_URL}`"
        )
        yield history, new_session_id
    except Exception as e:
        history[-1][1] = f"Unexpected error: {e}"
        yield history, new_session_id


def _session_label(session_id: Optional[str]) -> str:
    return f"Session: `{session_id[:8]}…`" if session_id else "No active session"


def create_gradio_interface():
    with gr.Blocks(title="arXiv Paper Curator — RAG Chat") as interface:
        session_state = gr.State(None)

        gr.Markdown(
            """
            # arXiv Paper Curator — RAG Chat

            Ask questions about machine learning and AI research papers from arXiv.
            Follow-up questions are supported — the model remembers your conversation within the session.
            Start a **New Session** to clear the history and begin a fresh conversation.
            """
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=520,
            layout="bubble",
            buttons=["copy", "copy_all"],
            placeholder="Your conversation will appear here.",
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="Your question",
                placeholder="What are transformers in machine learning?",
                lines=2,
                max_lines=5,
                scale=4,
                show_label=False,
                container=False,
            )
            with gr.Column(scale=1, min_width=130):
                submit_btn = gr.Button("Send", variant="primary", size="lg")
                new_session_btn = gr.Button("New Session", variant="secondary")

        session_indicator = gr.Markdown(_session_label(None))

        with gr.Accordion("Advanced Options", open=False):
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Chunks to retrieve",
                info="More chunks = more context but slower generation",
            )
            use_hybrid = gr.Checkbox(
                value=True,
                label="Hybrid search (BM25 + vector embeddings)",
                info="Usually gives better results than keyword-only search",
            )
            model_choice = gr.Dropdown(
                choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"],
                value=DEFAULT_MODEL,
                label="LLM Model",
                info="Larger models may give better answers but are slower",
            )
            categories_input = gr.Textbox(
                label="arXiv Categories (optional)",
                placeholder="cs.AI, cs.LG, cs.CL",
                info="Comma-separated. Leave empty for all categories",
            )

        gr.Examples(
            examples=[
                ["What are transformers in machine learning?"],
                ["How do convolutional neural networks work?"],
                ["What is the attention mechanism in deep learning?"],
                ["Explain reinforcement learning from human feedback (RLHF)"],
                ["What are the latest developments in NLP?"],
            ],
            inputs=[msg_input],
        )

        gr.Markdown(
            """
            ---
            **Note**: Make sure the RAG API server is running at `http://localhost:8000`.

            **Categories**: cs.AI · cs.LG · cs.CL · cs.CV · cs.NE · stat.ML
            """
        )

        # ── Event handlers ──────────────────────────────────────────────────

        async def respond(message, history, sid, top_k, use_hybrid, model, cats):
            if not message.strip():
                yield history, sid, _session_label(sid), ""
                return
            async for hist, new_sid in _stream_chat_turn(
                message, history, sid, top_k, use_hybrid, model, cats
            ):
                yield hist, new_sid, _session_label(new_sid), ""

        def new_session():
            return [], None, _session_label(None)

        shared_inputs = [msg_input, chatbot, session_state, top_k, use_hybrid, model_choice, categories_input]
        shared_outputs = [chatbot, session_state, session_indicator, msg_input]

        submit_btn.click(fn=respond, inputs=shared_inputs, outputs=shared_outputs)
        msg_input.submit(fn=respond, inputs=shared_inputs, outputs=shared_outputs)
        new_session_btn.click(fn=new_session, outputs=[chatbot, session_state, session_indicator])

    return interface


def main():
    print("Starting arXiv Paper Curator Gradio Interface...")
    print(f"API Base URL: {API_BASE_URL}")

    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
