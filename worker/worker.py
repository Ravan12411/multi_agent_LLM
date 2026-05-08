import asyncio
import os
import time

import asyncpg

DB_URL = os.environ.get("DATABASE_URL", "postgresql://user:pass@db:5432/agentdb")


async def process_pending_jobs():
    """Background worker that monitors and processes queued jobs."""
    pool = None
    while not pool:
        try:
            pool = await asyncpg.create_pool(DB_URL)
        except Exception:
            print("Waiting for DB...")
            await asyncio.sleep(3)

    print("Worker started. Monitoring jobs table...")

    while True:
        try:
            async with pool.acquire() as conn:
                # Check for any stale processing jobs (> 5 min)
                stale = await conn.fetch("""
                    SELECT job_id, query FROM jobs
                    WHERE status = 'processing'
                    AND created_at < NOW() - INTERVAL '5 minutes'
                """)
                for job in stale:
                    print(f"Marking stale job {job['job_id']} as failed")
                    await conn.execute(
                        "UPDATE jobs SET status='failed' WHERE job_id=$1",
                        job["job_id"]
                    )

                # Log worker heartbeat
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS worker_heartbeat (
                        id SERIAL PRIMARY KEY,
                        ts TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute("INSERT INTO worker_heartbeat DEFAULT VALUES")

        except Exception as e:
            print(f"Worker error: {e}")

        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(process_pending_jobs())
