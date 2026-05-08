"""
critique.py - Reviews every other agent's output with structured confidence scores per claim.
"""
import json
import time
import google.generativeai as genai

from ..core.schema import (
    AgentID, AgentOutput, ClaimScore, EventType, SharedContext
)
from ..core import logger
from ..core.context_manager import consume_tokens, estimate_tokens, maybe_compress

CRITIQUE_PROMPT = """You are a critique agent. Review the provided agent outputs with extreme precision.

For each output, you must:
1. Break it into individual claims
2. Score each claim's confidence from 0.0 to 1.0
3. Flag claims you disagree with (flagged: true) with a specific reason
4. Do NOT flag the entire output - flag SPECIFIC TEXT SPANS only

Return JSON:
{
  "reviews": [
    {
      "agent_id": "agent_name",
      "claim_scores": [
        {
          "claim_text": "exact span of text from the output",
          "confidence": 0.85,
          "flagged": false,
          "reason": null
        },
        {
          "claim_text": "another claim that seems wrong",
          "confidence": 0.3,
          "flagged": true,
          "reason": "This contradicts chunk_web_001 which states..."
        }
      ],
      "overall_confidence": 0.75
    }
  ],
  "critical_issues": ["list of serious problems found, if any"]
}

Return ONLY valid JSON."""


async def run(ctx: SharedContext, model) -> AgentOutput:
    start = time.time()

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.CRITIQUE.value,
        event_type=EventType.AGENT_START,
    )

    # Gather all agent outputs to critique
    outputs_to_review = []
    for aid, output in ctx.agent_outputs.items():
        outputs_to_review.append(f"=== {aid} ===\n{output.content[:500]}")

    outputs_text = "\n\n".join(outputs_to_review)
    prompt = f"Query: {ctx.original_query}\n\nAgent outputs to critique:\n{outputs_text}"

    compressed = await maybe_compress(ctx, AgentID.CRITIQUE.value, prompt)
    tokens = estimate_tokens(compressed)
    consume_tokens(ctx, AgentID.CRITIQUE.value, tokens)

    try:
        response = model.generate_content(
            f"{CRITIQUE_PROMPT}\n\n{compressed}",
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        reviews = parsed.get("reviews", [])
        issues = parsed.get("critical_issues", [])

        # Update claim scores on each agent's output
        all_claim_scores = []
        for review in reviews:
            aid = review.get("agent_id", "")
            claims = [ClaimScore(**c) for c in review.get("claim_scores", [])]
            all_claim_scores.extend(claims)
            if aid in ctx.agent_outputs:
                ctx.agent_outputs[aid].claim_scores = claims

        content = f"Critiqued {len(reviews)} agents.\nIssues found: {'; '.join(issues) if issues else 'None'}"

    except Exception as e:
        content = f"Critique failed: {str(e)}"
        all_claim_scores = []

    out_tokens = estimate_tokens(content)
    consume_tokens(ctx, AgentID.CRITIQUE.value, out_tokens)
    latency = (time.time() - start) * 1000

    output = AgentOutput(
        agent_id=AgentID.CRITIQUE,
        content=content,
        claim_scores=all_claim_scores,
        token_count=tokens + out_tokens
    )
    ctx.agent_outputs[AgentID.CRITIQUE.value] = output

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.CRITIQUE.value,
        event_type=EventType.AGENT_END,
        output_text=content,
        latency_ms=latency,
        token_count=tokens + out_tokens
    )

    return output
