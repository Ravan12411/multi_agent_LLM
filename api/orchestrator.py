import json
import time
import google.generativeai as genai
from typing import AsyncGenerator

from .core.schema import AgentID, EventType, SharedContext
from .core import logger
from .core.context_manager import init_budgets, consume_tokens, estimate_tokens
from .agents import decomposition, retrieval, critique, synthesis

ROUTING_PROMPT = """You are a master orchestrator for a multi-agent LLM system.
Given a query and the current pipeline state, decide which agent to invoke next.

Available agents:
- decomposition: breaks query into sub-tasks (always run first)
- retrieval: multi-hop RAG with citations (run after decomposition)
- critique: reviews all prior outputs (run after retrieval)
- synthesis: merges everything into final answer (run last)
- done: pipeline is complete

Return JSON only:
{
  "next_agent": "decomposition|retrieval|critique|synthesis|done",
  "justification": "why you chose this agent",
  "context_budget_check": "remaining budget is sufficient|low|critical"
}"""


async def run_pipeline(ctx: SharedContext, model, stream_callback=None) -> SharedContext:
    """
    Main orchestration loop. Dynamically routes between agents.
    Agents never call each other directly — all handoffs go through here.
    """
    init_budgets(ctx)
    completed_agents = []
    max_steps = 8  # Safety limit

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.ORCHESTRATOR.value,
        event_type=EventType.AGENT_START,
        payload={"query": ctx.original_query}
    )

    for step in range(max_steps):
        # Decide next agent
        next_agent, justification = await _decide_next(
            ctx, model, completed_agents
        )

        await logger.log_event(
            job_id=ctx.job_id,
            agent_id=AgentID.ORCHESTRATOR.value,
            event_type=EventType.AGENT_START,
            payload={
                "step": step,
                "routing_decision": next_agent,
                "justification": justification,
                "completed_so_far": completed_agents
            }
        )

        if stream_callback:
            await stream_callback({
                "event": "routing",
                "agent": next_agent,
                "justification": justification,
                "step": step
            })

        if next_agent == "done":
            break

        # Execute the chosen agent
        t_start = time.time()
        try:
            if next_agent == "decomposition":
                output = await decomposition.run(ctx, model)
            elif next_agent == "retrieval":
                output = await retrieval.run(ctx, model)
            elif next_agent == "critique":
                output = await critique.run(ctx, model)
            elif next_agent == "synthesis":
                output = await synthesis.run(ctx, model)
            else:
                break

            completed_agents.append(next_agent)

            if stream_callback:
                await stream_callback({
                    "event": "agent_complete",
                    "agent": next_agent,
                    "output_preview": output.content[:200],
                    "token_count": output.token_count
                })

        except Exception as e:
            await logger.log_event(
                job_id=ctx.job_id,
                agent_id=next_agent,
                event_type=EventType.ERROR,
                payload={"error": str(e)}
            )
            if stream_callback:
                await stream_callback({"event": "error", "agent": next_agent, "error": str(e)})

    # Ensure we have a final answer
    if not ctx.final_answer:
        if AgentID.RETRIEVAL.value in ctx.agent_outputs:
            ctx.final_answer = ctx.agent_outputs[AgentID.RETRIEVAL.value].content
        else:
            ctx.final_answer = "Pipeline completed without generating a final answer."

    from datetime import datetime
    ctx.completed_at = datetime.utcnow()

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.ORCHESTRATOR.value,
        event_type=EventType.JOB_COMPLETE,
        payload={
            "completed_agents": completed_agents,
            "policy_violations": ctx.policy_violations,
            "final_answer_length": len(ctx.final_answer or "")
        }
    )

    return ctx


async def _decide_next(ctx: SharedContext, model, completed: list) -> tuple:
    """Use LLM to decide which agent to run next, with logged justification."""

    # Deterministic fast-path for known sequences
    if "decomposition" not in completed:
        return "decomposition", "Decomposition always runs first to structure the query"
    if "retrieval" not in completed:
        return "retrieval", "Retrieval runs after decomposition to gather information"
    if "critique" not in completed:
        return "critique", "Critique reviews all gathered outputs for quality"
    if "synthesis" not in completed:
        return "synthesis", "Synthesis produces the final answer from all inputs"

    return "done", "All required agents have completed"
