# Session Architecture

## Request Flow

```mermaid
flowchart TD
    GU(Gradio UI)
    GU -->|"query + optional session_id"| API[FastAPI]
    API --> Q{"session_id provided?"}

    Q -->|yes| HL[Load history from Redis]
    Q -->|no| CC{"Exact cache hit?"}

    CC -->|yes| RET[Return cached response]
    RET --> GU

    CC -->|no| EMB[Jina embed query]
    HL --> EMB

    EMB --> OS[OpenSearch hybrid search]
    OS --> PB[Build prompt with history]
    PB --> LLM[Ollama LLM]
    LLM --> SH[Persist turn to Redis]

    SH --> Q2{"session_id provided?"}
    Q2 -->|no| SC[Store in exact-match cache]
    Q2 -->|yes| RESP[Return response]
    SC --> RESP
    RESP --> GU
```

## Stateless vs Session

| | No `session_id` | With `session_id` |
|---|:---:|:---:|
| Load history | — | ✓ |
| Check cache | ✓ | — |
| History in prompt | — | ✓ |
| Save turn | ✓ | ✓ |
| Store in cache | ✓ | — |
| `session_id` in response | new UUID | echoed |

## Redis Keys

| Key | TTL | Notes |
|---|---|---|
| `exact_cache:{sha256[:16]}` | 6 h | Keyed on query + model + top_k + use_hybrid + categories |
| `session:{id}:history` | 24 h | Resets on each turn. Stores last 5 turns (10 messages) |
