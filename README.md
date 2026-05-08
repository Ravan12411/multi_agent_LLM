# Multi-Agent LLM Orchestration and Evaluation System

A production-grade, containerized multi-agent system with a self-improving evaluation loop, dynamic tool orchestration, and adversarial robustness testing.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (Port 8000)                   │
│                  5 Endpoints + SSE Stream                │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   ORCHESTRATOR      │  ← Dynamic routing, no hardcoded chains
          │   (mediates all     │    Logs every routing decision + justification
          │    agent handoffs)  │
          └──┬──────┬──────┬───┘
             │      │      │
    ┌────────▼─┐ ┌──▼───┐ ┌▼────────┐ ┌──────────┐
    │DECOMPOSE │ │ RAG  │ │CRITIQUE │ │SYNTHESIS │
    │          │ │      │ │         │ │          │
    │Breaks    │ │Multi-│ │Per-claim│ │Merges all│
    │query into│ │hop   │ │scoring  │ │resolves  │
    │dep graph │ │+cite │ │flagging │ │contradict│
    └──────────┘ └──┬───┘ └─────────┘ └──────────┘
                    │
          ┌─────────▼──────────┐
          │      4 TOOLS        │
          │ web_search          │
          │ code_execute        │
          │ db_lookup (NL→SQL)  │
          │ self_reflect        │
          └─────────────────────┘
                    │
          ┌─────────▼──────────┐
          │   SHARED CONTEXT   │  ← All agents read/write here
          │   (Pydantic schema)│    Orchestrator mediates all access
          └─────────────────────┘
                    │
          ┌─────────▼──────────┐
          │    PostgreSQL       │
          │  logs, jobs,        │
          │  eval_runs,         │
          │  prompt_rewrites    │
          └─────────────────────┘
```

## Quick Start (5 minutes)

### Prerequisites
- Docker Desktop installed and running
- A Google Gemini API key (free at https://aistudio.google.com)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/multi-agent-llm
cd multi-agent-llm

# 2. Set your API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 3. Start everything
docker compose up --build

# 4. Test it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

API docs available at: http://localhost:8000/docs

Log viewer (Adminer) at: http://localhost:8080
- Server: db, User: user, Password: pass, DB: agentdb

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Submit query → SSE stream of real-time agent activity |
| GET | `/trace/{job_id}` | Full execution trace for any completed job |
| GET | `/eval/latest` | Latest eval run summary by category + dimension |
| POST | `/eval/run` | Trigger full 15-case evaluation harness |
| POST | `/eval/rewrite/review` | Approve or reject a meta-agent prompt rewrite |
| POST | `/eval/rerun-failed` | Re-eval failed cases with latest approved prompts |
| GET | `/eval/rewrite/pending` | List pending prompt rewrites awaiting review |

## Agent Decision Boundaries

### Orchestrator
- Runs first on every query
- Decides agent invocation order at runtime via structured reasoning
- All inter-agent communication passes through shared context — agents never call each other
- Logs every routing decision with justification
- Hard limit: 8 steps per pipeline to prevent infinite loops

### Decomposition Agent
- Always runs first (pre-retrieval)
- Breaks queries into 2-4 typed sub-tasks: `retrieval | reasoning | code | lookup`
- Produces explicit dependency graphs — dependent tasks cannot execute until prerequisites resolve
- Falls back to single-task if parsing fails

### Retrieval Agent (RAG)
- Requires decomposition output to complete first
- Performs minimum 2-hop reasoning (Hop 1 → Hop 2 builds on Hop 1)
- Calls web_search + db_lookup tools with up to 2 retries each
- Every claim must cite a chunk_id using `[cite:chunk_id]` format
- Returns structured hop reasoning in output

### Critique Agent
- Runs after retrieval
- Reviews EVERY other agent's output — not the pipeline as a whole
- Assigns per-claim confidence scores (0.0–1.0)
- Flags specific text spans with disagreement reasons
- Updates claim_scores on each agent's output in shared context

### Synthesis Agent
- Runs last, after critique
- Resolves contradictions flagged by critique
- Produces provenance map: every output sentence traced to source agent + chunk
- If synthesis fails, falls back to retrieval output

### Compression Agent (inline)
- Triggers automatically when any agent exceeds context budget
- Lossless for structured data (JSON, citations, scores)
- Lossy (30% retention) for conversational prose

### Meta Agent
- Reads eval failure cases after each eval run
- Identifies the worst-performing agent+dimension combination
- Proposes a structured prompt rewrite with diff and justification
- Rewrite is NEVER auto-applied — requires human approval via API

## Evaluation Harness

15 test cases across 3 categories:

| Category | Count | Tests |
|----------|-------|-------|
| Baseline | 5 | Known correct answers, keyword matching |
| Ambiguous | 5 | Underspecified inputs testing decomposition quality |
| Adversarial | 5 | Prompt injections, false premises, contradiction triggers |

6 scoring dimensions (all produce numeric score + written justification):

| Dimension | What it measures |
|-----------|-----------------|
| answer_correctness | Keyword coverage vs expected answer |
| citation_accuracy | Valid chunk_id citations in final answer |
| contradiction_resolution | Flagged claims vs provenance map entries |
| tool_efficiency | Accepted calls / total calls, penalizes retries |
| context_budget_compliance | Policy violations (budget overruns) |
| critique_agreement_rate | Synthesis confidence vs flagged claim count |

Every eval run stored in PostgreSQL with full reproducibility. Re-runs produce diff-able output.

## Self-Improving Prompt Loop

```
eval run → meta agent reads failures
        → identifies worst agent+dimension
        → proposes rewrite with structured diff
        → stores as "pending" (NOT auto-applied)
        → human reviews via POST /eval/rewrite/review
        → if approved: re-runs only failed cases
        → stores delta score + timestamp
        → all decisions queryable in prompt_rewrites table
```

## Known Limitations

1. **Web search is a stub** — returns generated results, not real internet data. In production, replace with SerpAPI or Tavily.

2. **NL→SQL is keyword-based** — the db_lookup tool uses heuristic keyword matching, not a real LLM SQL generator. Works for the sample dataset.

3. **Context compression is naive** — the compression agent uses line-based heuristics. A production system would use a dedicated summarization model.

4. **Single-process orchestrator** — the orchestrator runs in-process. High concurrency would require a proper task queue (Celery, RQ).

5. **Gemini rate limits** — free tier Gemini 1.5 Flash has RPM limits. Rapid eval runs may hit rate limits.

6. **Self-reflection is session-scoped** — the self_reflect tool only sees outputs within the current job's shared context, not across sessions.

## What I Would Build Next

- Replace web search stub with real API (Tavily/SerpAPI)
- Add ChromaDB for persistent vector storage across sessions
- Implement proper LLM-based NL→SQL with schema injection
- Add Prometheus + Grafana for real-time observability dashboards
- Horizontal scaling with Redis-backed job queue
- A/B testing framework for prompt rewrites across eval cohorts
- Human-in-the-loop review UI for prompt rewrite approvals
