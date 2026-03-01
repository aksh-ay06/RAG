# Multi-Turn Dialogue Architecture

## Request Flow

```mermaid
flowchart TD
    Client(["Client\n(browser / curl / Gradio)"])
    Client -->|"POST /api/v1/ask\nor /api/v1/stream\n{ query, session_id? }"| Router

    subgraph Router["FastAPI Router  (src/routers/ask.py)"]
        direction TB
        Resolve["Resolve session_id\n(use provided OR uuid4().hex)"]
        Branch{session_id\nprovided?}
        Resolve --> Branch
    end

    subgraph Redis["Redis"]
        direction TB
        ExactCache[("exact_cache:{hash}\nTTL: 6 h")]
        SessionStore[("session:{id}:history\nTTL: 24 h\nlast 10 msgs kept")]
    end

    Branch -->|"Yes → fetch prior context"| SessionStore
    SessionStore -->|"[ ] or [{role,content}, …]"| Search

    Branch -->|"No → check exact match"| ExactCache
    ExactCache -->|"cache miss"| Search
    ExactCache -->|"cache hit ✓"| CacheReturn["Stream / return\ncached answer"]
    CacheReturn --> Client

    subgraph Retrieval["Retrieval  (src/services/opensearch/client.py)"]
        Search["OpenSearch\nhybrid search\n(BM25 + vector)"]
    end

    Search -->|"top-k chunks"| PromptBuilder

    subgraph Generation["Generation  (src/services/ollama/)"]
        direction TB
        PromptBuilder["RAGPromptBuilder\n─────────────────\nsystem prompt\n[Conversation History]\n  User: …\n  Assistant: …\nContext from Papers\n  [1. arXiv:id] …\nQuestion"]
        Ollama["Ollama LLM\n(llama3.2 / configurable)"]
        PromptBuilder -->|"final prompt"| Ollama
    end

    Ollama -->|"answer text"| Persist

    subgraph Persist["Persist  (src/services/cache/client.py)"]
        direction LR
        AppendHistory["append_to_session_history\nsession:{id}:history"]
        StoreCache["store_response\nexact_cache:{hash}"]
        AppendHistory -. "always" .-> StoreCache
    end

    Persist -->|"session turn"| SessionStore
    Persist -->|"stateless only"| ExactCache

    Persist -->|"{ answer, session_id, sources, … }"| Client
```

## Redis Key Spaces

```mermaid
erDiagram
    EXACT_CACHE {
        string key        "exact_cache:{sha256[:16]}"
        json   value      "AskResponse JSON"
        int    ttl_hours  "6 (configurable)"
    }
    SESSION_HISTORY {
        string key        "session:{uuid4_hex}:history"
        json   value      "[{role, content}, …]  max 10 msgs"
        int    ttl_hours  "24 (configurable, resets on each turn)"
    }
```

## Session vs Stateless Decision Matrix

| | **No `session_id`** | **With `session_id`** |
|---|---|---|
| Fetch history | ✗ | ✓ `get_session_history(id)` |
| Check exact-match cache | ✓ | ✗ |
| Inject history into prompt | ✗ | ✓ (if history non-empty) |
| Append turn to history | ✓ | ✓ |
| Store in exact-match cache | ✓ | ✗ |
| `session_id` in response | ✓ (new) | ✓ (echoed) |

## Component Map

```mermaid
graph LR
    subgraph Config["src/config.py"]
        RS["RedisSettings\n· ttl_hours = 6\n· session_ttl_hours = 24"]
    end

    subgraph Schemas["src/schemas/api/ask.py"]
        AskReq["AskRequest\n+ session_id?: str"]
        AskResp["AskResponse\n+ session_id?: str"]
    end

    subgraph Cache["src/services/cache/client.py"]
        CC["CacheClient\n· find_cached_response()\n· store_response()\n· get_session_history()\n· append_to_session_history()"]
    end

    subgraph Prompts["src/services/ollama/prompts.py"]
        PB["RAGPromptBuilder\n· create_rag_prompt(\n    query, chunks,\n    history=None\n  )"]
    end

    subgraph RouterFile["src/routers/ask.py"]
        AskEP["POST /ask"]
        StreamEP["POST /stream"]
    end

    RS --> CC
    AskReq --> AskEP
    AskReq --> StreamEP
    CC --> AskEP
    CC --> StreamEP
    PB --> AskEP
    PB --> StreamEP
    AskEP --> AskResp
    StreamEP --> AskResp
```
