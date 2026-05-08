import json
import time
import google.generativeai as genai

from ..core.schema import (
    AgentID, AgentOutput, ClaimScore, EventType, ProvenanceEntry, SharedContext
)
from ..core import logger
from ..core.context_manager import consume_tokens, estimate_tokens, maybe_compress

SYNTHESIS_PROMPT = """You are a synthesis agent. Your job is to merge outputs from all agents into a final, coherent answer.

You must:
1. Resolve any contradictions flagged by the critique agent
2. Produce a final answer where EVERY sentence traces back to a source agent and chunk
3. Return a JSON provenance map

Return JSON:
{
  "final_answer": "Your complete synthesized answer here. Each sentence should be coherent and contradiction-free.",
  "provenance_map": [
    {
      "sentence": "exact sentence from final_answer",
      "source_agent": "retrieval|decomposition|critique",
      "source_chunk_id": "chunk_web_001 or null"
    }
  ],
  "contradictions_resolved": [
    {
      "conflict": "description of the conflict",
      "resolution": "how you resolved it"
    }
  ],
  "confidence": 0.0-1.0
}

Return ONLY valid JSON."""


async def run(ctx: SharedContext, model) -> AgentOutput:
    start = time.time()

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.SYNTHESIS.value,
        event_type=EventType.AGENT_START,
    )

    # Gather all outputs including critique
    parts = []
    for aid, output in ctx.agent_outputs.items():
        flagged = [c for c in output.claim_scores if c.flagged]
        flag_note = f" [FLAGGED CLAIMS: {len(flagged)}]" if flagged else ""
        parts.append(f"=== {aid}{flag_note} ===\n{output.content[:600]}")

    # Include chunk info for provenance
    chunk_summary = "\n".join([
        f"{c.chunk_id}: {c.content[:100]}" for c in ctx.retrieved_chunks[:5]
    ])

    prompt = f"""Query: {ctx.original_query}

All agent outputs:
{chr(10).join(parts)}

Available chunks for citation:
{chunk_summary}

Synthesize a final answer resolving all contradictions with full provenance."""

    compressed = await maybe_compress(ctx, AgentID.SYNTHESIS.value, prompt)
    tokens = estimate_tokens(compressed)
    consume_tokens(ctx, AgentID.SYNTHESIS.value, tokens)

    try:
        response = model.generate_content(
            f"{SYNTHESIS_PROMPT}\n\n{compressed}",
            generation_config=genai.GenerationConfig(temperature=0.2)
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        final_answer = parsed.get("final_answer", "")
        provenance = parsed.get("provenance_map", [])
        resolutions = parsed.get("contradictions_resolved", [])
        confidence = parsed.get("confidence", 0.8)

        # Store provenance in context
        ctx.provenance_map = [
            ProvenanceEntry(
                sentence=p.get("sentence", ""),
                source_agent=AgentID(p.get("source_agent", "synthesis")) if p.get("source_agent") in [a.value for a in AgentID] else AgentID.SYNTHESIS,
                source_chunk_id=p.get("source_chunk_id")
            )
            for p in provenance
        ]

        ctx.final_answer = final_answer
        content = final_answer
        if resolutions:
            content += f"\n\n[Resolved {len(resolutions)} contradictions]"

        claim_scores = [ClaimScore(
            claim_text=final_answer[:100],
            confidence=confidence,
            flagged=confidence < 0.5
        )]

    except Exception as e:
        content = f"Synthesis failed: {str(e)}\nFallback: {ctx.agent_outputs.get('retrieval', AgentOutput(agent_id=AgentID.SYNTHESIS, content='No retrieval output')).content[:200]}"
        ctx.final_answer = content
        claim_scores = []

    out_tokens = estimate_tokens(content)
    consume_tokens(ctx, AgentID.SYNTHESIS.value, out_tokens)
    latency = (time.time() - start) * 1000

    output = AgentOutput(
        agent_id=AgentID.SYNTHESIS,
        content=content,
        metadata={"provenance_entries": len(ctx.provenance_map)},
        claim_scores=claim_scores,
        token_count=tokens + out_tokens
    )
    ctx.agent_outputs[AgentID.SYNTHESIS.value] = output

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.SYNTHESIS.value,
        event_type=EventType.AGENT_END,
        output_text=content,
        latency_ms=latency,
        token_count=tokens + out_tokens
    )

    return output
