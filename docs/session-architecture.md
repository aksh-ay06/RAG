# Session Architecture

## Request Flow

```mermaid
flowchart TD
    Client([Client])
    Client -->|query| Router[Router]
    Router --> Q{Session?}

    subgraph Redis
        Hist[History]
        Cache[Cache]
        Save[Save Turn]
    end

    Q -->|yes| Hist
    Q -->|no|  Cache

    Cache -->|hit|  Client
    Cache -->|miss| Search[OpenSearch]
    Hist --> Search

    Search --> Prompt[Prompt Builder]
    Prompt --> LLM[Ollama]
    LLM --> Save
    Save --> Client
```

## Stateless vs Session

| | No `session_id` | With `session_id` |
|---|:---:|:---:|
| Load history | — | ✓ |
| Check cache | ✓ | — |
| History in prompt | — | ✓ |
| Save turn | ✓ | ✓ |
| Store in cache | ✓ | — |
| `session_id` in response | new | echoed |

## Redis Keys

| Key | TTL |
|---|---|
| `exact_cache:{hash}` | 6 h |
| `session:{id}:history` | 24 h (resets each turn) |
