import asyncio

async def web_search(query: str) -> dict:
    """Search the web for the given query. Returns a dict of URLs and relevance scores."""
    # Stub implementation
    return {
        "results": [
            {"source_id": "mock_web_1", "snippet": f"Mock result for: {query}", "url": "https://example.com/1", "relevance_score": 0.9},
            {"source_id": "mock_web_2", "snippet": "Additional context for the query", "url": "https://example.com/2", "relevance_score": 0.75}
        ],
        "failure_type": None
    }
