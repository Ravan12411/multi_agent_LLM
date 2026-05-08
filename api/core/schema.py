from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid
from datetime import datetime


class AgentID(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DECOMPOSITION = "decomposition"
    RETRIEVAL = "retrieval"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    COMPRESSION = "compression"
    META = "meta"


class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT_BUDGET = "context_budget"
    POLICY_VIOLATION = "policy_violation"
    STREAM_TOKEN = "stream_token"
    JOB_COMPLETE = "job_complete"
    ERROR = "error"


class ToolFailureType(str, Enum):
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"
    MALFORMED_INPUT = "malformed_input"
    EXECUTION_ERROR = "execution_error"


class SubTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    task_type: str  # "retrieval", "reasoning", "code", "lookup"
    dependencies: List[str] = []  # list of task_ids this depends on
    resolved: bool = False
    result: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    relevance_score: float


class ClaimScore(BaseModel):
    claim_text: str
    confidence: float  # 0.0 to 1.0
    flagged: bool
    reason: Optional[str] = None


class AgentOutput(BaseModel):
    agent_id: AgentID
    content: str
    metadata: Dict[str, Any] = {}
    claim_scores: List[ClaimScore] = []
    cited_chunks: List[str] = []  # chunk_ids
    token_count: int = 0


class ToolCall(BaseModel):
    tool_name: str
    input: Dict[str, Any]
    output: Optional[Any] = None
    latency_ms: float = 0.0
    accepted: bool = True
    failure_type: Optional[ToolFailureType] = None
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: AgentID
    source_chunk_id: Optional[str] = None


class SharedContext(BaseModel):
    """The single shared object all agents read/write through the orchestrator."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str
    sub_tasks: List[SubTask] = []
    agent_outputs: Dict[str, AgentOutput] = {}  # keyed by agent_id
    tool_calls: List[ToolCall] = []
    retrieved_chunks: List[Chunk] = []
    provenance_map: List[ProvenanceEntry] = []
    final_answer: Optional[str] = None
    token_budgets: Dict[str, int] = {}   # agent_id -> max tokens
    token_used: Dict[str, int] = {}      # agent_id -> tokens used so far
    policy_violations: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class LogEntry(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    event_type: EventType
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    latency_ms: float = 0.0
    token_count: int = 0
    payload: Dict[str, Any] = {}
    policy_violation: Optional[str] = None


class EvalScore(BaseModel):
    test_id: str
    query: str
    category: str  # "baseline", "ambiguous", "adversarial"
    answer_correctness: float
    citation_accuracy: float
    contradiction_resolution: float
    tool_efficiency: float
    context_budget_compliance: float
    critique_agreement_rate: float
    justifications: Dict[str, str] = {}
    final_answer: str = ""


class EvalRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scores: List[EvalScore] = []
    overall_score: float = 0.0


class PromptRewrite(BaseModel):
    rewrite_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    original_prompt: str
    proposed_prompt: str
    diff_summary: str
    justification: str
    worst_dimension: str
    worst_score: float
    status: str = "pending"  # pending, approved, rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    delta_score: Optional[float] = None
