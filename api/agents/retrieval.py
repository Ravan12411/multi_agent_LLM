import json
import time
from typing import List
import google.generativeai as genai

from ..core.schema import (
    AgentID, AgentOutput, Chunk, ClaimScore, EventType, SharedContext, ToolCall
)
from ..core import logger
from ..core.context_manager import consume_tokens, estimate_tokens, maybe_compress
from ..tools.web_search import web_search
from ..tools.db_lookup import db_lookup

SYSTEM_PROMPT = """You are a retrieval-augmented reasoning agent. You perform multi-hop reasoning across retrieved information.

You will receive retrieved chunks. You MUST:
1. Perform at least TWO hops of reasoning - each hop builds on the previous
2. For each claim in your answer, cite which chunk_id it came from using format [cite:chunk_id]
3. Return a JSON object:
{
  "answer": "your multi-hop answer with [cite:chunk_id] inline citations",
  "hop_1": "what you learned from first retrieval hop",
  "hop_2": "what you learned by combining hop_1 with additional chunks", 
  "cited_chunks": ["chunk_id_1", "chunk_id_2"],
  "confidence": 0.0-1.0
}

Single-hop retrieval is NOT sufficient. You must demonstrate chained reasoning.
Return ONLY valid JSON."""


async def run(ctx: SharedContext, model) -> AgentOutput:
    start = time.time()

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.RETRIEVAL.value,
        event_type=EventType.AGENT_START,
        payload={"query": ctx.original_query}
    )

    # --- Tool Call 1: Web Search ---
    search_result = await _call_tool_with_retry(
        ctx, "web_search", {"query": ctx.original_query}, web_search, ctx.original_query
    )

    # --- Tool Call 2: DB Lookup ---
    db_result = await _call_tool_with_retry(
        ctx, "db_lookup", {"query": ctx.original_query}, db_lookup, ctx.original_query
    )

    # Build chunks from tool results
    chunks = _build_chunks(search_result, db_result)
    ctx.retrieved_chunks.extend(chunks)

    # Build context for LLM
    chunk_text = "\n\n".join([
        f"[{c.chunk_id}] (relevance: {c.relevance_score:.2f})\nSource: {c.source}\n{c.content}"
        for c in chunks
    ])

    prompt = f"""Query: {ctx.original_query}

Sub-tasks context:
{chr(10).join([f'- {t.description}' for t in ctx.sub_tasks])}

Retrieved chunks:
{chunk_text}

Perform multi-hop reasoning and answer with citations."""

    compressed = await maybe_compress(ctx, AgentID.RETRIEVAL.value, prompt)
    tokens = estimate_tokens(compressed)
    consume_tokens(ctx, AgentID.RETRIEVAL.value, tokens)

    try:
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\n{compressed}",
            generation_config=genai.GenerationConfig(temperature=0.3)
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        answer = parsed.get("answer", "")
        cited = parsed.get("cited_chunks", [])
        confidence = parsed.get("confidence", 0.7)
        hop1 = parsed.get("hop_1", "")
        hop2 = parsed.get("hop_2", "")
        content = f"{answer}\n\nHop 1: {hop1}\nHop 2: {hop2}"

        claim_scores = [ClaimScore(
            claim_text=answer[:100],
            confidence=confidence,
            flagged=confidence < 0.6
        )]

    except Exception as e:
        content = f"Retrieval reasoning failed: {str(e)}"
        cited = []
        claim_scores = []

    out_tokens = estimate_tokens(content)
    consume_tokens(ctx, AgentID.RETRIEVAL.value, out_tokens)
    latency = (time.time() - start) * 1000

    output = AgentOutput(
        agent_id=AgentID.RETRIEVAL,
        content=content,
        metadata={"chunks_retrieved": len(chunks)},
        claim_scores=claim_scores,
        cited_chunks=cited,
        token_count=tokens + out_tokens
    )
    ctx.agent_outputs[AgentID.RETRIEVAL.value] = output

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.RETRIEVAL.value,
        event_type=EventType.AGENT_END,
        input_text=prompt,
        output_text=content,
        latency_ms=latency,
        token_count=tokens + out_tokens
    )

    return output


async def _call_tool_with_retry(ctx, tool_name, tool_input, tool_fn, query, max_retries=2):
    """Call a tool with up to max_retries retries on failure."""
    for attempt in range(max_retries + 1):
        t_start = time.time()
        result = await tool_fn(query)
        latency = (time.time() - t_start) * 1000
        accepted = result.get("failure_type") is None

        tool_call = ToolCall(
            tool_name=tool_name,
            input=tool_input,
            output=result,
            latency_ms=latency,
            accepted=accepted,
            failure_type=result.get("failure_type"),
            retry_count=attempt
        )
        ctx.tool_calls.append(tool_call)

        await logger.log_event(
            job_id=ctx.job_id,
            agent_id=AgentID.RETRIEVAL.value,
            event_type=EventType.TOOL_CALL,
            payload={"tool": tool_name, "attempt": attempt, "accepted": accepted, "latency_ms": latency}
        )

        if accepted:
            return result

        # Modify input for retry
        query = f"{query} detailed explanation"

    return result  # Return last result even if failed


def _build_chunks(search_result: dict, db_result: dict) -> List[Chunk]:
    chunks = []

    for r in search_result.get("results", []):
        chunks.append(Chunk(
            chunk_id=r.get("source_id", f"chunk_web_{len(chunks)}"),
            content=r.get("snippet", ""),
            source=r.get("url", "web"),
            relevance_score=r.get("relevance_score", 0.7)
        ))

    for i, row in enumerate(db_result.get("rows", [])):
        chunks.append(Chunk(
            chunk_id=f"chunk_db_{i:03d}",
            content=f"{row.get('topic', '')}: {row.get('description', '')} (score: {row.get('score', 0)})",
            source=f"database/{row.get('category', 'unknown')}",
            relevance_score=row.get("score", 0.7)
        ))

    return chunks
