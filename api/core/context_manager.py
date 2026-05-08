from typing import Optional
from .schema import SharedContext, AgentID, EventType
from . import logger


# Default token budgets per agent
DEFAULT_BUDGETS = {
    AgentID.ORCHESTRATOR:   4000,
    AgentID.DECOMPOSITION:  2000,
    AgentID.RETRIEVAL:      3000,
    AgentID.CRITIQUE:       3000,
    AgentID.SYNTHESIS:      4000,
    AgentID.COMPRESSION:    2000,
    AgentID.META:           2000,
}


def init_budgets(ctx: SharedContext):
    """Initialize token budgets for all agents."""
    for agent_id, budget in DEFAULT_BUDGETS.items():
        ctx.token_budgets[agent_id.value] = budget
        ctx.token_used[agent_id.value] = 0


def check_budget(ctx: SharedContext, agent_id: str) -> int:
    """Returns remaining token budget for agent. Negative means over budget."""
    budget = ctx.token_budgets.get(agent_id, 2000)
    used = ctx.token_used.get(agent_id, 0)
    return budget - used


def consume_tokens(ctx: SharedContext, agent_id: str, tokens: int) -> bool:
    """
    Record token consumption. Returns True if within budget, False if exceeded.
    Logs a policy violation if exceeded.
    """
    if agent_id not in ctx.token_used:
        ctx.token_used[agent_id] = 0

    ctx.token_used[agent_id] += tokens
    remaining = check_budget(ctx, agent_id)

    if remaining < 0:
        violation = f"Agent {agent_id} exceeded context budget by {abs(remaining)} tokens"
        ctx.policy_violations.append(violation)
        return False
    return True


def estimate_tokens(text: str) -> int:
    """Rough token estimator: ~4 chars per token."""
    return max(1, len(text) // 4)


async def maybe_compress(ctx: SharedContext, agent_id: str, content: str) -> str:
    """
    If agent is over budget, compress older conversational content.
    Structured data (JSON-like) is preserved; filler prose is summarized.
    """
    remaining = check_budget(ctx, agent_id)
    needed = estimate_tokens(content)

    if needed <= remaining:
        return content  # No compression needed

    # Log that compression is happening
    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=agent_id,
        event_type=EventType.CONTEXT_BUDGET,
        payload={
            "action": "compression_triggered",
            "remaining_budget": remaining,
            "needed": needed,
        }
    )

    # Simple compression: keep first 20% and last 80% of content by token weight
    # Structured lines (JSON, citations) are always preserved
    lines = content.split('\n')
    structured = [l for l in lines if any(c in l for c in ['{', '[', 'chunk_', 'score:', 'cite:'])]
    prose = [l for l in lines if l not in structured]

    # Keep all structured, compress prose to 30% of original
    keep_prose_count = max(1, len(prose) // 3)
    compressed_prose = prose[:keep_prose_count] + ["... [compressed] ..."] + prose[-keep_prose_count:]

    compressed = '\n'.join(structured + compressed_prose)
    return compressed
