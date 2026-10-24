"""API 层运行时行为单测（不依赖 Redis/Chroma/LLM）。

用桩件替换 lifespan 装配的全局组件，直接调用路由函数：
  - /chat 的锁-运行-事件-工单-审计完整落库链路；
  - 会话并发 409；
  - 升级工单的状态机与重复裁决 409；
  - revise 修订意见回流下一次运行的闭环；
  - 运行取消（条件终态 + 同事务审计）与启动僵尸对账；
  - Prometheus 业务指标增量；
  - API Key 认证依赖（经 TestClient 中间层触发，不跑 lifespan）。
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main as api_main
from agents.agent_orchestrator import AgentType
from api.main import ChatRequest, HandoffResolveInput, RunCancelInput
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.runtime import RuntimeStore


class FakeMemory:
    async def get_context(self, user_id, conv_id, query=""):
        return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "")

    async def add_message(self, user_id, conv_id, role, content):
        pass

    async def update_profile(self, user_id, conv_id):
        pass

    async def close(self):
        pass


class FakeOrchestrator:
    def __init__(self, escalated=False):
        self.escalated = escalated
        self.run_calls = []

    async def recognize_intent(self, message, history=None):
        return SimpleNamespace(
            intent=IntentCategory.BILLING,
            intent_group="card",
            confidence=0.9,
            entities={"amount": ["99 元"]},
            urgency=UrgencyLevel.HIGH,
            source_scores={"llm": 0.7},
        )

    async def run(self, req):
        self.run_calls.append(req)
        return SimpleNamespace(
            request_id="req-1",
            response="已收到您的账单疑问，将转人工核验。",
            agent_type=AgentType.BILLING,
            intent=IntentCategory.BILLING,
            escalated=self.escalated,
            latency_ms=12.0,
            agent_types=[AgentType.BILLING],
            primary_agent=AgentType.BILLING,
            supporting_agents=[],
            tools_used=["search_knowledge_base"],
            tool_traces=[{
                "agent_type": "billing",
                "tool_name": "search_knowledge_base",
                "tool_use_id": "tu_1",
                "input": {"query": "账单"},
                "success": True,
                "latency_ms": 3.2,
            }],
            routing_reason="intent=billing",
            routing_confidence=0.9,
        )


@pytest.fixture()
def runtime():
    store = RuntimeStore(":memory:")
    yield store
    store.close()


@pytest.fixture()
def wired(runtime, monkeypatch):
    """把三个全局组件替换为真实 store + 桩件，/chat 路由即可脱离外部依赖运行。"""
    orch = FakeOrchestrator()
    monkeypatch.setattr(api_main, "_runtime", runtime)
    monkeypatch.setattr(api_main, "_memory", FakeMemory())
    monkeypatch.setattr(api_main, "_orchestrator", orch)
    # 画像更新是异步旁路任务，测试里直接关闭该协程避免悬挂任务告警
    monkeypatch.setattr(api_main.asyncio, "create_task", lambda coro: coro.close())
    return SimpleNamespace(store=runtime, orch=orch)


def test_chat_persists_run_events_messages_and_releases_lock(wired):
    resp = asyncio.run(api_main.chat(ChatRequest(message="我的账单有问题", user_id="u1")))

    assert resp.run_id
    run = wired.store.get_run(resp.run_id)
    assert run["status"] == "completed"

    event_types = [e["type"] for e in wired.store.list_events(resp.run_id)]
    assert event_types == ["message.user", "route", "tool", "message.assistant"]

    transcript = wired.store.list_messages(resp.conv_id)
    assert [(m["role"], m["content"]) for m in transcript] == [
        ("user", "我的账单有问题"),
        ("assistant", "已收到您的账单疑问，将转人工核验。"),
    ]

    # 执行锁已释放：同会话可以立即发起下一次运行
    assert wired.store.acquire_turn_lock(resp.conv_id, "next-run")

    # 审计已留痕
    entries = wired.store.list_audit(resource_type="run", resource_id=resp.run_id)
    assert entries[0]["action"] == "chat.turn"
    assert entries[0]["details"]["tools_used"] == ["search_knowledge_base"]


def test_chat_concurrent_request_on_same_conversation_returns_409(wired):
    wired.store.acquire_turn_lock("conv-busy", "ghost-run")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_main.chat(ChatRequest(message="再来一条", user_id="u1", conv_id="conv-busy")))
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "conversation_run_active"
    assert exc.value.detail["active_run_id"] == "ghost-run"
    # 409 之后锁仍然属于原持有者
    assert wired.store.active_lock("conv-busy")["run_id"] == "ghost-run"


def test_chat_failure_marks_run_failed_and_audits(wired, monkeypatch):
    async def boom(req):
        raise RuntimeError("LLM 超时")

    monkeypatch.setattr(wired.orch, "run", boom)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_main.chat(ChatRequest(message="触发失败", user_id="u1")))
    assert exc.value.status_code == 500

    audit = wired.store.list_audit(action="chat.failed")
    assert len(audit) == 1
    run_id = audit[0]["run_id"]
    assert wired.store.get_run(run_id)["status"] == "failed"
    # 失败后锁已释放，同会话可以继续发起下一次运行
    assert wired.store.acquire_turn_lock(audit[0]["conv_id"], "next-run")


def test_escalated_run_creates_pending_ticket_and_resolves(wired):
    wired.orch.escalated = True
    resp = asyncio.run(api_main.chat(ChatRequest(message="你们催收逼死人", user_id="u1")))

    assert resp.escalated is True
    assert resp.ticket_id
    ticket = wired.store.get_handoff(resp.ticket_id)
    assert ticket["status"] == "pending"
    assert ticket["kind"] == "crisis"  # 危机关键词命中 → 更高优先级
    assert ticket["priority"] == 1

    resolved = asyncio.run(api_main.resolve_handoff(
        resp.ticket_id, HandoffResolveInput(decision="accept", resolver="human-01")
    ))
    assert resolved["status"] == "accepted"

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_main.resolve_handoff(
            resp.ticket_id, HandoffResolveInput(decision="reject")
        ))
    assert exc.value.status_code == 409
    assert exc.value.detail["current_status"] == "accepted"


def test_handoff_revise_writes_back_to_run_events(wired):
    wired.orch.escalated = True
    resp = asyncio.run(api_main.chat(ChatRequest(message="帮我处理退款", user_id="u1")))

    resolved = asyncio.run(api_main.resolve_handoff(
        resp.ticket_id, HandoffResolveInput(decision="revise", note="先补充卡号后四位")
    ))
    assert resolved["status"] == "pending"
    assert resolved["revision_count"] == 1

    events = wired.store.list_events(resp.run_id)
    revisions = [e for e in events if e["type"] == "handoff.revision_requested"]
    assert revisions and revisions[0]["payload"]["note"] == "先补充卡号后四位"


def test_revision_note_reaches_next_run_in_same_conversation(wired):
    """闭环：AI 建议 → 坐席 revise → 同会话下一次运行自动携带修订要求。"""
    wired.orch.escalated = True
    first = asyncio.run(api_main.chat(ChatRequest(message="帮我处理退款", user_id="u1")))
    asyncio.run(api_main.resolve_handoff(
        first.ticket_id, HandoffResolveInput(decision="revise", note="先核验卡号后四位再答复")
    ))

    wired.orch.escalated = False
    second = asyncio.run(api_main.chat(
        ChatRequest(message="那再处理一次", user_id="u1", conv_id=first.conv_id)
    ))

    # 响应与事件流都带修订上下文，坐席要求进入模型输入
    assert second.revision_applied["ticket_id"] == first.ticket_id
    assert second.revision_applied["revision_note"] == "先核验卡号后四位再答复"
    assert "先核验卡号后四位再答复" in wired.orch.run_calls[-1].context
    event_types = [e["type"] for e in wired.store.list_events(second.run_id)]
    assert "handoff.revision_applied" in event_types

    # 工单接受后，修订上下文停止注入
    asyncio.run(api_main.resolve_handoff(
        first.ticket_id, HandoffResolveInput(decision="accept")
    ))
    third = asyncio.run(api_main.chat(
        ChatRequest(message="再确认一下", user_id="u1", conv_id=first.conv_id)
    ))
    assert third.revision_applied is None


def test_cancel_run_endpoint_and_reconcile(wired):
    resp = asyncio.run(api_main.chat(ChatRequest(message="我的账单有问题", user_id="u1")))

    # 已终结的运行不可取消：409 且不覆盖已有终态
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_main.cancel_run(resp.run_id, RunCancelInput(reason="误操作")))
    assert exc.value.status_code == 409
    assert exc.value.detail["current_status"] == "completed"
    assert wired.store.get_run(resp.run_id)["status"] == "completed"

    # 崩溃残留的 running 可取消，审计同事务
    wired.store.create_run("run-zombie", "conv-zombie", "u1")
    cancelled = asyncio.run(api_main.cancel_run("run-zombie", RunCancelInput(reason="坐席终止")))
    assert cancelled["run"]["status"] == "cancelled"
    audit = wired.store.list_audit(action="run.cancelled", resource_id="run-zombie")
    assert audit and audit[0]["details"]["reason"] == "坐席终止"

    # 启动对账：过期 running 归档为 cancelled（同一条 store 路径）
    wired.store.create_run("run-stale", "conv-stale", "u1")
    reconciled = wired.store.reconcile_stale_runs(max_age_s=-1)
    assert [r["run_id"] for r in reconciled] == ["run-stale"]
    assert wired.store.get_run("run-stale")["status"] == "cancelled"


def test_business_metrics_counters_increment(wired):
    from prometheus_client import REGISTRY

    def sample(name, **labels):
        return REGISTRY.get_sample_value(name, labels) or 0.0

    turns_before = sample("creditlife_chat_turns_total", intent_group="card", escalated="false")
    conflicts_before = sample("creditlife_turn_lock_conflicts_total")

    asyncio.run(api_main.chat(ChatRequest(message="我的账单有问题", user_id="u1")))
    assert sample(
        "creditlife_chat_turns_total", intent_group="card", escalated="false"
    ) == turns_before + 1

    wired.store.acquire_turn_lock("conv-busy", "ghost-run")
    with pytest.raises(HTTPException):
        asyncio.run(api_main.chat(
            ChatRequest(message="再来一条", user_id="u1", conv_id="conv-busy")
        ))
    assert sample("creditlife_turn_lock_conflicts_total") == conflicts_before + 1


def test_api_key_dependency_blocks_unauthenticated_requests(monkeypatch):
    monkeypatch.setattr(api_main, "_orchestrator", None)
    monkeypatch.setenv("CREDITLIFE_API_KEY", "secret-123")

    client = TestClient(api_main.app)  # 不进入 lifespan，仅验证依赖与鉴权层
    resp_no_key = client.post("/chat", json={"message": "你好"})
    assert resp_no_key.status_code == 401

    resp_wrong = client.post("/chat", json={"message": "你好"}, headers={"X-API-Key": "wrong"})
    assert resp_wrong.status_code == 401

    resp_ok = client.post("/chat", json={"message": "你好"}, headers={"X-API-Key": "secret-123"})
    # 通过鉴权后进入路由，因组件未初始化返回 503（而非 401）
    assert resp_ok.status_code == 503


def test_api_key_unset_keeps_local_demo_open(monkeypatch):
    monkeypatch.delenv("CREDITLIFE_API_KEY", raising=False)
    monkeypatch.setattr(api_main, "_orchestrator", None)

    client = TestClient(api_main.app)
    assert client.post("/chat", json={"message": "你好"}).status_code == 503
    assert client.get("/health").status_code == 503  # 组件未装配，但未被鉴权拦截
