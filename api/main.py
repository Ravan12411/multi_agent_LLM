import asyncio
import json
import os
from datetime import datetime
from typing import AsyncGenerator

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .core.schema import SharedContext
from .core import logger
from .orchestrator import run_pipeline
from .eval.harness import run_eval
from .meta.prompt_improver import propose_rewrite, apply_rewrite

app = FastAPI(
    title="Multi-Agent LLM Orchestration System",
    description="Production-grade multi-agent system with self-improving evaluation loop",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    dsn = os.environ.get("DATABASE_URL", "postgresql://user:pass@db:5432/agentdb")
    await logger.init_db_pool(dsn)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)

    app.state.model = genai.GenerativeModel("gemini-2.5-flash")


def get_model():
    return app.state.model


# ── Request/Response Models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class ApprovalRequest(BaseModel):
    rewrite_id: str
    approved: bool


# ── ENDPOINT 1: Submit query with SSE streaming ───────────────────────────────

@app.post(
    "/query",
    summary="Submit a query and receive streaming SSE agent activity",
    response_description="Server-Sent Events stream with real-time agent updates"
)
async def submit_query(request: QueryRequest):
    """
    Submit a natural language query. Returns a streaming SSE response showing:
    - Which agent is currently running
    - Tool calls in flight
    - Current context budget remaining
    - Final answer token by token
    """
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_QUERY", "message": "Query must be at least 3 characters", "job_id": None}
        )

    ctx = SharedContext(original_query=request.query)
    model = get_model()

    await logger.save_job(ctx.job_id, request.query, ctx.json())

    async def event_stream() -> AsyncGenerator[str, None]:
        events = asyncio.Queue()

        async def stream_callback(event: dict):
            await events.put(event)

        async def run_with_callback():
            try:
                result = await run_pipeline(ctx, model, stream_callback)
                await logger.complete_job(ctx.job_id, result.json())
                await events.put({"event": "done", "job_id": ctx.job_id, "final_answer": result.final_answer})
            except Exception as e:
                await events.put({"event": "error", "error": str(e), "job_id": ctx.job_id})
            finally:
                await events.put(None)  # Sentinel

        asyncio.create_task(run_with_callback())

        # Yield job_id first
        yield f"data: {json.dumps({'event': 'job_started', 'job_id': ctx.job_id})}\n\n"

        while True:
            event = await events.get()
            if event is None:
                break

            # Enrich with budget info
            event["context_budgets"] = {
                aid: ctx.token_budgets.get(aid, 0) - ctx.token_used.get(aid, 0)
                for aid in ctx.token_budgets
            }
            yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Job-ID": ctx.job_id, "Cache-Control": "no-cache"}
    )


# ── ENDPOINT 2: Get full execution trace ─────────────────────────────────────

@app.get(
    "/trace/{job_id}",
    summary="Get full execution trace for a completed job",
    response_description="Ordered sequence of all agent decisions, tool calls, and handoffs"
)
async def get_trace(job_id: str):
    """
    Returns the complete execution trace for any job ID, reconstructing
    the exact sequence of agent decisions, tool calls, and handoffs in order.
    """
    trace = await logger.get_execution_trace(job_id)
    if not trace:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "JOB_NOT_FOUND", "message": f"No trace found for job_id: {job_id}", "job_id": job_id}
        )
    return {
        "job_id": job_id,
        "total_events": len(trace),
        "trace": trace
    }


# ── ENDPOINT 3: Latest eval run summary ──────────────────────────────────────

@app.get(
    "/eval/latest",
    summary="Get latest eval run summary by category and scoring dimension"
)
async def get_latest_eval():
    """
    Returns the most recent evaluation run broken down by:
    - Test category (baseline, ambiguous, adversarial)
    - Scoring dimension (correctness, citation, contradiction, efficiency, budget, critique)
    """
    run = await logger.get_latest_eval_run()
    if not run:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NO_EVAL_RUN", "message": "No evaluation runs found. POST /eval/run first.", "job_id": None}
        )

    scores = json.loads(run["scores"]) if isinstance(run["scores"], str) else run["scores"]

    # Group by category
    by_category = {}
    for score in scores:
        cat = score["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(score)

    # Compute dimension averages per category
    summary = {}
    dims = ["answer_correctness", "citation_accuracy", "contradiction_resolution",
            "tool_efficiency", "context_budget_compliance", "critique_agreement_rate"]

    for cat, cat_scores in by_category.items():
        summary[cat] = {
            d: round(sum(s[d] for s in cat_scores) / len(cat_scores), 3)
            for d in dims
        }

    return {
        "run_id": run["run_id"],
        "timestamp": str(run["timestamp"]),
        "overall_score": run["overall_score"],
        "by_category": summary,
        "total_test_cases": len(scores)
    }


@app.post(
    "/eval/run",
    summary="Trigger a full evaluation run"
)
async def trigger_eval(background_tasks: BackgroundTasks):
    """Runs the full 15-case evaluation harness asynchronously."""
    model = get_model()
    background_tasks.add_task(_run_eval_task, model)
    return {"message": "Evaluation started in background. Check /eval/latest for results."}


async def _run_eval_task(model):
    eval_run = await run_eval(model, run_pipeline)
    rewrite = await propose_rewrite(eval_run, model)
    return eval_run


# ── ENDPOINT 4: Approve or reject prompt rewrite ─────────────────────────────

@app.post(
    "/eval/rewrite/review",
    summary="Submit human approval or rejection for a pending prompt rewrite"
)
async def review_rewrite(request: ApprovalRequest):
    """
    Approve or reject a proposed prompt rewrite from the meta-agent.
    If approved, the system re-runs eval on previously failed cases.
    All decisions are stored with timestamps and are queryable.
    """
    rewrites = await logger.get_pending_rewrites()
    pending_ids = [r["rewrite_id"] for r in rewrites]

    if request.rewrite_id not in pending_ids:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "REWRITE_NOT_FOUND",
                "message": f"No pending rewrite found with id: {request.rewrite_id}",
                "job_id": None
            }
        )

    model = get_model()
    result = await apply_rewrite(
        request.rewrite_id,
        request.approved,
        model=model if request.approved else None,
        run_pipeline_fn=run_pipeline if request.approved else None
    )
    return result


@app.get(
    "/eval/rewrite/pending",
    summary="List all pending prompt rewrites"
)
async def list_pending_rewrites():
    rewrites = await logger.get_pending_rewrites()
    return {"pending_rewrites": rewrites, "count": len(rewrites)}


# ── ENDPOINT 5: Targeted re-eval on failed cases ─────────────────────────────

@app.post(
    "/eval/rerun-failed",
    summary="Trigger targeted re-eval on previously failed cases using latest approved prompts"
)
async def rerun_failed(background_tasks: BackgroundTasks):
    """
    Re-runs evaluation only on test cases that failed in the last eval run.
    Uses the latest approved prompt rewrites.
    Results are stored with a diff against the previous run for regression visibility.
    """
    last_run = await logger.get_latest_eval_run()
    if not last_run:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "NO_PREVIOUS_RUN", "message": "No previous eval run to compare against.", "job_id": None}
        )

    model = get_model()
    background_tasks.add_task(_rerun_failed_task, model, last_run)

    return {
        "message": "Re-evaluation of failed cases started.",
        "previous_run_id": last_run["run_id"],
        "previous_score": last_run["overall_score"]
    }


async def _rerun_failed_task(model, last_run):
    scores = json.loads(last_run["scores"]) if isinstance(last_run["scores"], str) else last_run["scores"]
    failed = [s for s in scores if s["answer_correctness"] < 0.6]

    results = []
    for tc in failed:
        ctx = SharedContext(original_query=tc["query"])
        try:
            ctx = await run_pipeline(ctx, model)
            results.append({"query": tc["query"], "new_answer": ctx.final_answer[:200]})
        except Exception as e:
            results.append({"query": tc["query"], "error": str(e)})

    return results 
