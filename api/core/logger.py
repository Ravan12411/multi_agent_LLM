import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

import asyncpg

from .schema import EventType, LogEntry


_pool: Optional[asyncpg.Pool] = None


async def init_db_pool(dsn: str):
    global _pool
    _pool = await asyncpg.create_pool(dsn)
    await _create_tables()


async def _create_tables():
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                log_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                agent_id TEXT,
                event_type TEXT,
                input_hash TEXT,
                output_hash TEXT,
                latency_ms FLOAT,
                token_count INT,
                payload JSONB,
                policy_violation TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_logs_job_id ON logs(job_id);
            CREATE INDEX IF NOT EXISTS idx_logs_event_type ON logs(event_type);

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                context JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                scores JSONB,
                overall_score FLOAT
            );

            CREATE TABLE IF NOT EXISTS prompt_rewrites (
                rewrite_id TEXT PRIMARY KEY,
                agent_id TEXT,
                original_prompt TEXT,
                proposed_prompt TEXT,
                diff_summary TEXT,
                justification TEXT,
                worst_dimension TEXT,
                worst_score FLOAT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                reviewed_at TIMESTAMPTZ,
                delta_score FLOAT
            );
        """)


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12] if text else ""


async def log_event(
    job_id: str,
    agent_id: str,
    event_type: EventType,
    payload: Dict[str, Any] = {},
    input_text: str = "",
    output_text: str = "",
    latency_ms: float = 0.0,
    token_count: int = 0,
    policy_violation: Optional[str] = None,
):
    entry = LogEntry(
        job_id=job_id,
        agent_id=agent_id,
        event_type=event_type,
        input_hash=_hash(input_text),
        output_hash=_hash(output_text),
        latency_ms=latency_ms,
        token_count=token_count,
        payload=payload,
        policy_violation=policy_violation,
    )
    if _pool:
        async with _pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO logs (log_id, job_id, timestamp, agent_id, event_type,
                    input_hash, output_hash, latency_ms, token_count, payload, policy_violation)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            """,
                entry.log_id, entry.job_id, entry.timestamp, entry.agent_id,
                entry.event_type.value, entry.input_hash, entry.output_hash,
                entry.latency_ms, entry.token_count,
                json.dumps(entry.payload), entry.policy_violation
            )
    return entry


async def get_execution_trace(job_id: str):
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM logs WHERE job_id=$1 ORDER BY timestamp ASC
        """, job_id)
        return [dict(r) for r in rows]


async def save_job(job_id: str, query: str, context_json: str):
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO jobs (job_id, query, status, context)
            VALUES ($1, $2, 'processing', $3::jsonb)
            ON CONFLICT (job_id) DO UPDATE SET context=$3::jsonb
        """, job_id, query, context_json)


async def complete_job(job_id: str, context_json: str):
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            UPDATE jobs SET status='completed', context=$2::jsonb, completed_at=NOW()
            WHERE job_id=$1
        """, job_id, context_json)


async def save_eval_run(run_id: str, timestamp: datetime, scores_json: str, overall: float):
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO eval_runs (run_id, timestamp, scores, overall_score)
            VALUES ($1,$2,$3::jsonb,$4)
            ON CONFLICT (run_id) DO UPDATE SET scores=$3::jsonb, overall_score=$4
        """, run_id, timestamp, scores_json, overall)


async def save_prompt_rewrite(rewrite):
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO prompt_rewrites
            (rewrite_id, agent_id, original_prompt, proposed_prompt,
             diff_summary, justification, worst_dimension, worst_score, status, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """, rewrite.rewrite_id, rewrite.agent_id, rewrite.original_prompt,
            rewrite.proposed_prompt, rewrite.diff_summary, rewrite.justification,
            rewrite.worst_dimension, rewrite.worst_score, rewrite.status, rewrite.created_at)


async def get_pending_rewrites():
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM prompt_rewrites WHERE status='pending'")
        return [dict(r) for r in rows]


async def update_rewrite_status(rewrite_id: str, status: str, delta: Optional[float] = None):
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute("""
            UPDATE prompt_rewrites SET status=$2, reviewed_at=NOW(), delta_score=$3
            WHERE rewrite_id=$1
        """, rewrite_id, status, delta)


async def get_latest_eval_run():
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT 1"
        )
        return dict(row) if row else None
