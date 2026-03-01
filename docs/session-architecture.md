# Multi-Turn Dialogue Architecture

## How a request flows through the system

```mermaid
flowchart TD
    Client(["Client"])
    Client -->|"query + optional session_id"| Branch

    Branch{{"session_id\nprovided?"}}

    Branch -->|"Yes"| History[("Load history\nfrom Redis")]
    Branch -->|"No"| Cache[("Check\nexact-match cache")]

    Cache -->|"Hit"| Done
    Cache -->|"Miss"| Search

    History --> Search

    Search["Search OpenSearch\nfor relevant chunks"]
    Search --> Prompt

    Prompt["Build prompt\n— system instructions\n— conversation history ①\n— paper excerpts\n— question"]
    Prompt --> LLM

    LLM["Ollama generates answer"]
    LLM --> Save

    Save["Save turn to Redis history\nCache response ②"]
    Save --> Done

    Done(["Return answer + session_id"])
    Done --> Client
```

> ① History is only injected when a `session_id` was provided and prior turns exist.
> ② Response is only written to the exact-match cache for stateless (no `session_id`) requests.

---

## What changes between stateless and session requests

| | No `session_id` | With `session_id` |
|---|:---:|:---:|
| Load conversation history | — | ✓ |
| Check exact-match cache | ✓ | — |
| History injected into prompt | — | ✓ |
| Save turn to history | ✓ | ✓ |
| Write to exact-match cache | ✓ | — |
| `session_id` in response | new UUID | echoed back |

---

## Redis storage

| Key | Contains | Expires |
|---|---|---|
| `exact_cache:{hash}` | Cached response JSON | 6 h |
| `session:{id}:history` | Last 10 messages (5 turns) | 24 h, resets each turn |
