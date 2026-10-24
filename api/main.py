"""
CreditLife 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from core.config import CRISIS_KEYWORDS, DEFAULT_MODEL
from core.runtime import RUN_ACTIVE, HANDOFF_PENDING, RuntimeStore, require_api_key

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   CreditLife  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_runtime: Optional[RuntimeStore] = None

# 需要 API Key 的路由共用本依赖；/health、/metrics 保持开放给探针和抓取。
_PROTECTED = [Depends(require_api_key)]

# ── 业务指标（/metrics 暴露，进程默认指标之外的可观测性）─────────────────────
CHAT_TURNS = Counter(
    "creditlife_chat_turns_total", "完成的对话轮次", ["intent_group", "escalated"]
)
RUN_LATENCY = Histogram(
    "creditlife_run_latency_seconds", "对话运行端到端时延", ["agent_type"]
)
TURN_LOCK_CONFLICTS = Counter(
    "creditlife_turn_lock_conflicts_total", "会话级执行锁冲突（409）"
)
HANDOFF_RESOLUTIONS = Counter(
    "creditlife_handoff_resolutions_total", "人工确认门裁决", ["decision"]
)
RUN_CANCELLATIONS = Counter(
    "creditlife_run_cancellations_total", "运行取消（人工/对账）", ["source"]
)

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _runtime

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request, build_shared_rag_tools
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 运行时持久层：会话/运行/事件/工单/审计落库（SQLite 单文件，可换 PostgreSQL）
    runtime_db = os.getenv("CREDITLIFE_RUNTIME_DB", str(pathlib.Path(_ROOT) / "data" / "runtime.db"))
    if runtime_db != ":memory:":
        pathlib.Path(runtime_db).parent.mkdir(parents=True, exist_ok=True)
    _runtime = RuntimeStore(runtime_db)
    # 僵尸运行对账：上次进程在 create_run 与 finish_run 之间崩溃留下的
    # running 记录，启动时统一归档为 cancelled（审计同事务），保证看板上
    # 的 running 永远代表真的在跑。
    stale_runs = _runtime.reconcile_stale_runs(
        float(os.getenv("CREDITLIFE_RECONCILE_MAX_AGE_S", "600"))
    )
    for stale in stale_runs:
        RUN_CANCELLATIONS.labels(source="reconcile").inc()
    if stale_runs:
        logger.warning("启动对账：归档 %d 条僵尸运行 -> %s", len(stale_runs), [r["run_id"] for r in stale_runs])
    logger.info(f"运行时持久层: {runtime_db}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("CREDITLIFE_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("CREDITLIFE_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description=(
            "银行信用卡业务知识库检索（ChromaDB 向量检索）。"
            "凡涉及业务规则的事实性问题，必须先调用本工具、再基于检索结果回答，禁止凭记忆直接给出具体数字或政策口径。"
            "覆盖：息费规则（利息/违约金/年费/宽限期）、分期、还款、额度、争议交易、盗刷、卡片邮寄、"
            "办卡进度、电子回单、电子渠道故障、客服合规红线。"
            "纯寒暄、情绪安抚、明确要求转人工的请求无需调用。"
        ),
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器（EVAL_JUDGE_MODEL 可指定独立评测模型，默认与对话同模型）
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=os.getenv("EVAL_JUDGE_MODEL") or cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("CreditLife 已就绪")
    yield

    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    if _runtime is not None:
        _runtime.close()
    logger.info("CreditLife 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CreditLife 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

# CORS 来源由环境决定：默认放开便于本地演示，生产必须配置为明确域名列表。
_cors_origins = [o.strip() for o in os.getenv("CREDITLIFE_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    run_id:      str = ""
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    ticket_id:   Optional[str] = None
    revision_applied: Optional[Dict[str, Any]] = None
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"], dependencies=_PROTECTED)
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse, dependencies=_PROTECTED)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      会话执行锁 → 持久化用户消息 → 意图识别 → Agent 路由 → 执行
      → 事件/审计落库 →（升级时）创建待人工确认工单 → 记忆写入

    运行全程落库（runs + run_events）；同一会话并发请求返回 409，
    断线后可通过 GET /runs/{run_id}/events 增量回放。
    """
    if _orchestrator is None or _memory is None or _runtime is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    # 0. 会话级执行锁：token + TTL。拿不到说明另一条消息正在该会话内运行。
    lock_token = _runtime.acquire_turn_lock(conv_id, run_id)
    if lock_token is None:
        active = _runtime.active_lock(conv_id)
        TURN_LOCK_CONFLICTS.inc()
        raise HTTPException(
            409,
            detail={
                "error": "conversation_run_active",
                "message": "该会话正在处理上一条消息，请稍后重试或使用新会话。",
                "active_run_id": (active or {}).get("run_id", ""),
            },
        )

    ticket_id: Optional[str] = None
    try:
        # 1. 读取记忆上下文
        mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

        history = [
            {"role": m.role.value, "content": m.content}
            for m in mem_ctx.recent_messages[-5:]
        ] if mem_ctx.recent_messages else None

        intent_result = await _orchestrator.recognize_intent(req.message, history=history)

        # 修订意见回流：会话内有带修订意见的 pending 工单时，本轮运行自动
        # 携带最新一轮坐席要求——"AI 建议 → 人工 revise → AI 带要求重跑"闭环。
        revision = _runtime.latest_pending_revision(conv_id)
        full_context = mem_ctx.to_prompt_text()
        revision_applied: Optional[Dict[str, Any]] = None
        if revision:
            revision_applied = {
                "ticket_id": revision["ticket_id"],
                "revision_count": revision["revision_count"],
                "revision_note": revision["revision_note"],
            }
            full_context += (
                f"\n\n[人工坐席修订意见（工单 {revision['ticket_id']}，"
                f"第 {revision['revision_count']} 轮修订）]\n"
                f"{revision['revision_note']}\n本轮回复必须先满足上述修订要求。"
            )

        # 2. 运行建档：用户消息与运行事件先落库，模型执行只负责产生新事件
        _runtime.create_run(
            run_id, conv_id, req.user_id,
            intent=intent_result.intent.value if intent_result.intent else "other",
        )
        _runtime.append_message(conv_id, run_id, MsgRole.USER.value, req.message)
        _runtime.append_event(run_id, "message.user", {"content": req.message})
        if revision_applied:
            _runtime.append_event(run_id, "handoff.revision_applied", dict(revision_applied))

        orch_req = OrcReq(
            message=req.message,
            user_id=req.user_id,
            conv_id=conv_id,
            context=full_context,
            history=history,
            entities=intent_result.entities,
            intent=intent_result.intent,
            intent_group=intent_result.intent_group,
            urgency=intent_result.urgency,
            intent_confidence=intent_result.confidence,
        )

        # 3. 执行
        result = await _orchestrator.run(orch_req)

        # 4. 路由决策、工具调用、最终回复依次落库，形成可回放的运行事件流
        _runtime.append_event(run_id, "route", {
            "primary_agent": result.primary_agent.value if result.primary_agent else result.agent_type.value,
            "supporting_agents": [a.value for a in result.supporting_agents],
            "routing_reason": result.routing_reason,
            "routing_confidence": result.routing_confidence,
            "intent": intent_result.intent.value if intent_result.intent else "other",
            "intent_group": intent_result.intent_group,
        })
        for trace in result.tool_traces:
            _runtime.append_event(run_id, "tool", trace)
        _runtime.append_message(conv_id, run_id, MsgRole.ASSISTANT.value, result.response)
        _runtime.append_event(run_id, "message.assistant", {"content": result.response})

        # 5. 升级 → 生成待人工确认的处置工单（AI 建议默认不自动执行）
        if result.escalated:
            is_crisis = any(kw in req.message for kw in CRISIS_KEYWORDS)
            urgency_is_critical = intent_result.urgency is not None and intent_result.urgency.name == "CRITICAL"
            priority = 1 if (is_crisis or urgency_is_critical) else 2
            ticket_id = _runtime.create_handoff(
                run_id=run_id,
                conv_id=conv_id,
                user_id=req.user_id,
                kind="crisis" if is_crisis else "handoff",
                summary=result.response,
                draft={
                    "intent": intent_result.intent.value if intent_result.intent else "other",
                    "intent_group": intent_result.intent_group,
                    "urgency": intent_result.urgency.name if intent_result.urgency else None,
                    "entities": intent_result.entities,
                    "tools_used": result.tools_used,
                    "request_id": result.request_id,
                },
                priority=priority,
            )
            _runtime.append_event(run_id, "handoff.created", {"ticket_id": ticket_id})

        # 6. 条件终态更新：重复回调不会重复终结
        _runtime.finish_run(run_id, "completed", escalated=result.escalated, agent_type=result.agent_type.value)

        # 7. 业务指标：轮次、时延（详见 /metrics）
        CHAT_TURNS.labels(
            intent_group=intent_result.intent_group or "other",
            escalated=str(bool(result.escalated)).lower(),
        ).inc()
        RUN_LATENCY.labels(agent_type=result.agent_type.value).observe(result.latency_ms / 1000.0)

        # 7. 审计与业务同库留痕
        _runtime.audit(
            action="chat.turn",
            user_id=req.user_id,
            conv_id=conv_id,
            run_id=run_id,
            resource_type="run",
            resource_id=run_id,
            details={
                "intent": intent_result.intent.value if intent_result.intent else "other",
                "agent_type": result.agent_type.value,
                "escalated": result.escalated,
                "tools_used": result.tools_used,
                "latency_ms": round(result.latency_ms, 1),
                "ticket_id": ticket_id,
            },
        )

        # 8. 写入记忆
        await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
        await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

        # 9. 异步更新用户画像（不阻塞响应）
        asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

        return ChatResponse(
            conv_id=conv_id,
            run_id=run_id,
            request_id=result.request_id,
            response=result.response,
            intent=result.intent.value if result.intent else "other",
            intent_group=intent_result.intent_group,
            agent_type=result.agent_type.value,
            agent_types=[agent_type.value for agent_type in result.agent_types],
            primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
            supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
            tools_used=result.tools_used,
            routing_reason=result.routing_reason,
            routing_confidence=result.routing_confidence,
            escalated=result.escalated,
            ticket_id=ticket_id,
            revision_applied=revision_applied,
            latency_ms=round(result.latency_ms, 1),
            knowledge_used="search_knowledge_base" in result.tools_used,
            entities=intent_result.entities,
            intent_confidence=round(intent_result.confidence, 4),
            intent_source_scores=intent_result.source_scores,
        )
    except HTTPException:
        raise
    except Exception as ex:
        logger.exception("会话 %s 运行 %s 执行失败", conv_id, run_id)
        _runtime.append_event(run_id, "run.failed", {"error": str(ex)})
        _runtime.finish_run(run_id, "failed")
        _runtime.audit(
            action="chat.failed",
            user_id=req.user_id,
            conv_id=conv_id,
            run_id=run_id,
            resource_type="run",
            resource_id=run_id,
            details={"error": str(ex)},
        )
        raise HTTPException(500, "处理请求时出现问题，请稍后重试")
    finally:
        _runtime.release_turn_lock(conv_id, lock_token)


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


# ── 运行时查询：回放 / 工单 / 审计 ────────────────────────────────────────────

@app.get("/runs/{run_id}", tags=["Runtime"], dependencies=_PROTECTED)
async def get_run(run_id: str):
    """查询单次运行的档案：状态、意图、事件数。"""
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    run = _runtime.get_run(run_id)
    if run is None:
        raise HTTPException(404, "运行不存在")
    events = _runtime.list_events(run_id)
    return {"run": run, "event_count": len(events), "last_seq": events[-1]["seq"] if events else 0}


@app.get("/runs/{run_id}/events", tags=["Runtime"], dependencies=_PROTECTED)
async def replay_run_events(run_id: str, after_seq: int = 0):
    """增量回放一次运行的事件流（断线恢复入口）。

    客户端记录已消费的最大 seq，重连后带 after_seq 续读，
    不重复消费也不漏事件。运行中断时已产生的事件仍然可读。
    """
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    run = _runtime.get_run(run_id)
    if run is None:
        raise HTTPException(404, "运行不存在")
    events = _runtime.list_events(run_id, after_seq=after_seq)
    return {"run_id": run_id, "status": run["status"], "after_seq": after_seq, "events": events}


class RunCancelInput(BaseModel):
    """取消运行输入。用于坐席/运维终止卡住的运行或崩溃残留的僵尸运行。"""
    reason: str = ""
    actor: str = "agent-console"


@app.post("/runs/{run_id}/cancel", tags=["Runtime"], dependencies=_PROTECTED)
async def cancel_run(run_id: str, body: RunCancelInput):
    """取消一次运行：条件更新 running → cancelled，审计同事务落库。

    running 的三个终态（completed / failed / cancelled）由此全部可达；
    已终结的运行重复取消返回 409（幂等 no-op，不覆盖已有终态）。
    服务启动时的僵尸对账走同一条 store.cancel_run 路径，审计口径一致。
    """
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    run = _runtime.get_run(run_id)
    if run is None:
        raise HTTPException(404, "运行不存在")
    if not _runtime.cancel_run(run_id, reason=body.reason, actor=body.actor):
        raise HTTPException(
            409,
            detail={"error": "run_not_running", "current_status": run["status"]},
        )
    RUN_CANCELLATIONS.labels(source="api").inc()
    return {"run": _runtime.get_run(run_id)}


@app.get("/conversations/{conv_id}/messages", tags=["Runtime"], dependencies=_PROTECTED)
async def get_conversation_messages(conv_id: str, limit: int = 100):
    """会话全文记录（已脱敏），供人工坐席接手时阅读上下文。"""
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    return {"conv_id": conv_id, "messages": _runtime.list_messages(conv_id, limit=limit)}


class HandoffResolveInput(BaseModel):
    """人工裁决输入。decision: accept（接管处理）/ reject（误报关闭）/ revise（补充要求，保持待处理）。"""
    decision: str
    note: str = ""
    resolver: str = "agent-console"


@app.get("/handoffs", tags=["Handoff"], dependencies=_PROTECTED)
async def list_handoffs(status: Optional[str] = "pending", limit: int = 50):
    """人工坐席工作队列：默认列出待处理的升级工单，按优先级和时间排序。"""
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    return {"items": _runtime.list_handoffs(status=status, limit=limit)}


@app.get("/handoffs/{ticket_id}", tags=["Handoff"], dependencies=_PROTECTED)
async def get_handoff(ticket_id: str):
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    ticket = _runtime.get_handoff(ticket_id)
    if ticket is None:
        raise HTTPException(404, "工单不存在")
    return ticket


@app.post("/handoffs/{ticket_id}/resolve", tags=["Handoff"], dependencies=_PROTECTED)
async def resolve_handoff(ticket_id: str, body: HandoffResolveInput):
    """人工裁决升级工单（人工确认门）。

    裁决串行在 pending 状态上：并发或重复提交只有一个生效，其余返回 409，
    不会重复执行业务。revise 保留原建议单并把修订意见回写原运行事件流。
    """
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    ticket = _runtime.get_handoff(ticket_id)
    if ticket is None:
        raise HTTPException(404, "工单不存在")
    resolved = _runtime.resolve_handoff(
        ticket_id,
        body.decision,
        note=body.note,
        resolver=body.resolver or "agent-console",
    )
    if resolved is None:
        raise HTTPException(
            409,
            detail={
                "error": "handoff_already_resolved",
                "message": "该工单已被处理，请刷新后查看最新状态。",
                "current_status": ticket["status"],
            },
        )
    HANDOFF_RESOLUTIONS.labels(decision=body.decision).inc()
    if body.decision == "revise":
        _runtime.append_event(ticket["run_id"], "handoff.revision_requested", {
            "ticket_id": ticket_id,
            "note": body.note,
            "revision_count": resolved["revision_count"],
        })
    return resolved


@app.get("/audit", tags=["Runtime"], dependencies=_PROTECTED)
async def list_audit(action: str = "", resource_type: str = "", resource_id: str = "", limit: int = 100):
    """审计流水查询：与业务写入同事务产生，支撑事后追责与合规检查。"""
    if _runtime is None:
        raise HTTPException(503, "服务未就绪")
    return {"items": _runtime.list_audit(action=action, resource_type=resource_type, resource_id=resource_id, limit=limit)}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口：进程默认指标 + creditlife_* 业务指标
    （轮次/时延/锁冲突/工单裁决/运行取消）。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search", dependencies=_PROTECTED)
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None
    expected_behavior: Optional[str] = None  # 合规/对抗用例的预期安全行为，透传给 LLM Judge


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。

    save_baseline: 是否把本次结果写入回归基线存档（EVAL_BASELINE_PATH）。
    默认 False——零星试跑不污染基线；只有完整评测集跑完才显式置 True。
    """
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    save_baseline: bool = False


@app.post("/knowledge/add", tags=["知识库"], dependencies=_PROTECTED)
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([{"title": d.title, "content": d.content} for d in body.documents])
    _tool_manager.invalidate_cache("knowledge_search")  # 知识库已更新，清除检索缓存避免 TTL 内返回旧结果
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"], dependencies=_PROTECTED)
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = await kb.add_documents_async(docs)
    _tool_manager.invalidate_cache("knowledge_search")  # 知识库已更新，清除检索缓存
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": await kb.doc_count_async()}


@app.post("/eval/run", dependencies=_PROTECTED)
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        save_baseline=bool(body.save_baseline) if body else False,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("CreditLife CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("CREDITLIFE_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("CREDITLIFE_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nCreditLife [{result.agent_type.value}]: {result.response}\n")

    await mem.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
