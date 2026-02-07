from fastapi import APIRouter
from src.schemas.ask import AskRequest, AskResponse, PaperSource

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:

    mock_sources = [
        PaperSource(
            arxiv_id="2401.00001",
            title="Mock Paper: Introduction to AI Research",
            authors=["John Doe", "Jane Smith"],
            abstract_preview="This is a mock abstract for testing purposes in initial testing...",
        ),
        PaperSource(
            arxiv_id="2401.00002",
            title="Mock Paper: Advanced Machine Learning Techniques",
            authors=["Alice Johnson", "Bob Wilson"],
            abstract_preview="Another mock abstract demonstrating the API structure...",
        ),
    ]

    return AskResponse(
        answer="This is a mock response for initial testing. Real search functionality will be implemented in later phases.",
        sources=mock_sources,
    )