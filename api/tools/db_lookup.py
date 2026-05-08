import os
import asyncio

async def db_lookup(query: str) -> dict:
    """Execute a mock DB lookup returning rows for the Retrieval agent."""
    query_lower = query.lower()
    if any(forbidden in query_lower for forbidden in ["insert", "update", "delete", "drop", "alter"]):
        return {"failure_type": "security_violation", "rows": []}
        
    try:
        # Returning mock rows directly. In a real system this translates the query to SQL and executes it.
        return {
            "rows": [
                {"topic": "DB record 1", "description": f"Internal data relating to {query}", "category": "knowledge_base", "score": 0.85}
            ],
            "failure_type": None
        }
    except Exception as e:
        return {"failure_type": "database_error", "rows": [], "error": str(e)}
