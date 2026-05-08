import json
import time
from typing import List
import google.generativeai as genai

from ..core.schema import (
    AgentID, AgentOutput, ClaimScore, EventType, SharedContext, SubTask
)
from ..core import logger
from ..core.context_manager import consume_tokens, estimate_tokens, maybe_compress

SYSTEM_PROMPT = """You are a decomposition agent. Your job is to break ambiguous queries into typed sub-tasks with explicit dependency graphs.

For each query, return a JSON object with this exact structure:
{
  "sub_tasks": [
    {
      "task_id": "t1",
      "description": "specific task description",
      "task_type": "retrieval|reasoning|code|lookup",
      "dependencies": []
    },
    {
      "task_id": "t2", 
      "description": "another task",
      "task_type": "reasoning",
      "dependencies": ["t1"]
    }
  ],
  "reasoning": "why you decomposed this way"
}

Rules:
- Tasks with dependencies must NOT execute before their dependencies resolve
- task_type must be one of: retrieval, reasoning, code, lookup
- Always produce 2-4 sub-tasks minimum
- Be specific, not vague
Return ONLY valid JSON, no markdown, no explanation outside the JSON."""


async def run(ctx: SharedContext, model) -> AgentOutput:
    start = time.time()

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.DECOMPOSITION.value,
        event_type=EventType.AGENT_START,
        payload={"query": ctx.original_query}
    )

    prompt = f"Query to decompose: {ctx.original_query}"
    compressed = await maybe_compress(ctx, AgentID.DECOMPOSITION.value, prompt)
    tokens = estimate_tokens(compressed)
    consume_tokens(ctx, AgentID.DECOMPOSITION.value, tokens)

    try:
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\n{compressed}",
            generation_config=genai.GenerationConfig(temperature=0.2)
        )
        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        sub_tasks = []
        for t in parsed.get("sub_tasks", []):
            sub_tasks.append(SubTask(
                task_id=t["task_id"],
                description=t["description"],
                task_type=t["task_type"],
                dependencies=t.get("dependencies", [])
            ))

        ctx.sub_tasks = sub_tasks
        reasoning = parsed.get("reasoning", "")
        content = f"Decomposed into {len(sub_tasks)} sub-tasks.\n{reasoning}"

        claim_scores = [
            ClaimScore(
                claim_text=f"Task {t.task_id}: {t.description[:60]}",
                confidence=0.9,
                flagged=False
            ) for t in sub_tasks
        ]

    except Exception as e:
        content = f"Decomposition failed: {str(e)}. Using single-task fallback."
        ctx.sub_tasks = [SubTask(
            task_id="t1",
            description=ctx.original_query,
            task_type="reasoning",
            dependencies=[]
        )]
        claim_scores = []

    latency = (time.time() - start) * 1000
    out_tokens = estimate_tokens(content)
    consume_tokens(ctx, AgentID.DECOMPOSITION.value, out_tokens)

    output = AgentOutput(
        agent_id=AgentID.DECOMPOSITION,
        content=content,
        metadata={"sub_task_count": len(ctx.sub_tasks)},
        claim_scores=claim_scores,
        token_count=tokens + out_tokens
    )
    ctx.agent_outputs[AgentID.DECOMPOSITION.value] = output

    await logger.log_event(
        job_id=ctx.job_id,
        agent_id=AgentID.DECOMPOSITION.value,
        event_type=EventType.AGENT_END,
        input_text=prompt,
        output_text=content,
        latency_ms=latency,
        token_count=tokens + out_tokens
    )

    return output
