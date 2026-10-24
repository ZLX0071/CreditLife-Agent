"""运行时持久层单测。

覆盖六条核心不变量：
  1. 会话级 turn lock 互斥、同运行重入、TTL 自动过期；
  2. 运行终态条件更新幂等（重复终结返回 False 而不是覆盖），含取消与僵尸对账；
  3. 事件流 seq 单调、支持 after_seq 增量回放；
  4. 工单裁决串行在 pending 上，审计与业务同事务；
  5. revise 的修订意见只挂在有修订的 pending 工单上，可被下一次运行取回；
  6. 取消运行同事务留审计。
"""

import pytest
from uuid import uuid4

from core.runtime import RuntimeStore
from core.runtime.masking import mask_pii


@pytest.fixture()
def store():
    store = RuntimeStore(":memory:")
    yield store
    store.close()


# ── turn lock ─────────────────────────────────────────────────────────────────

def test_turn_lock_is_exclusive_until_released(store):
    token = store.acquire_turn_lock("conv-1", "run-1")
    assert token
    assert store.acquire_turn_lock("conv-1", "run-2") is None
    assert store.active_lock("conv-1")["run_id"] == "run-1"

    assert store.release_turn_lock("conv-1", token) is True
    assert store.acquire_turn_lock("conv-1", "run-2")


def test_turn_lock_allows_reentry_for_same_run(store):
    token = store.acquire_turn_lock("conv-1", "run-1")
    assert store.acquire_turn_lock("conv-1", "run-1") == token


def test_turn_lock_expires_after_ttl(store):
    store.acquire_turn_lock("conv-1", "run-1", ttl_s=-1)
    assert store.active_lock("conv-1") is None
    assert store.acquire_turn_lock("conv-1", "run-2")


def test_heartbeat_only_works_for_current_owner(store):
    token = store.acquire_turn_lock("conv-1", "run-1")
    assert store.heartbeat_turn_lock("conv-1", token) is True
    assert store.heartbeat_turn_lock("conv-1", "wrong-token") is False


# ── 运行生命周期 ───────────────────────────────────────────────────────────────

def test_run_terminal_update_is_idempotent(store):
    store.create_run("run-1", "conv-1", "u1", intent="billing")
    assert store.get_run("run-1")["status"] == "running"

    assert store.finish_run("run-1", "completed", escalated=True, agent_type="billing") is True
    assert store.finish_run("run-1", "failed") is False
    run = store.get_run("run-1")
    assert run["status"] == "completed"
    assert run["escalated"] == 1
    assert run["completed_at"]


def test_events_are_monotonic_and_replayable(store):
    for i in range(3):
        store.append_event("run-1", "tool", {"i": i})

    all_events = store.list_events("run-1")
    assert [e["seq"] for e in all_events] == [1, 2, 3]
    assert [e["payload"]["i"] for e in store.list_events("run-1", after_seq=1)] == [1, 2]
    assert store.list_events("run-1", after_seq=3) == []


def test_list_runs_orders_by_recency(store):
    store.create_run("run-1", "conv-1", "u1")
    store.create_run("run-2", "conv-1", "u1")
    runs = store.list_runs("conv-1")
    assert [r["run_id"] for r in runs] == ["run-2", "run-1"]


def test_cancel_run_is_conditional_and_audited(store):
    store.create_run("run-1", "conv-1", "u1")

    assert store.cancel_run("run-1", reason="坐席终止", actor="human-01") is True
    run = store.get_run("run-1")
    assert run["status"] == "cancelled"
    assert run["completed_at"]

    # 已终结的运行不可再次取消/终结：条件更新幂等
    assert store.cancel_run("run-1") is False
    assert store.finish_run("run-1", "failed") is False

    entries = store.list_audit(action="run.cancelled", resource_id="run-1")
    assert len(entries) == 1
    assert entries[0]["details"]["reason"] == "坐席终止"
    assert entries[0]["details"]["actor"] == "human-01"


def test_reconcile_archives_only_stale_running_runs(store):
    store.create_run("run-fresh", "conv-1", "u1")
    store.create_run("run-stale", "conv-2", "u1")
    store.create_run("run-old", "conv-3", "u1")
    store.finish_run("run-old", "completed")

    # 负数 max_age 把所有 running 视为过期；已终结的 run-old 不受影响
    reconciled = store.reconcile_stale_runs(max_age_s=-1)
    assert sorted(r["run_id"] for r in reconciled) == ["run-fresh", "run-stale"]
    assert store.get_run("run-fresh")["status"] == "cancelled"
    assert store.get_run("run-stale")["status"] == "cancelled"
    assert store.get_run("run-old")["status"] == "completed"
    # 对账走 cancel_run 路径，审计口径一致
    assert len(store.list_audit(action="run.cancelled")) == 2


# ── 人工确认门 ────────────────────────────────────────────────────────────────

def make_ticket(store, summary="需要人工核验减免申请"):
    run_id = f"run-{uuid4().hex[:8]}"
    store.create_run(run_id, "conv-1", "u1")
    return store.create_handoff(
        run_id=run_id, conv_id="conv-1", user_id="u1",
        kind="handoff", summary=summary, draft={"intent": "refund"}, priority=2,
    )


def test_handoff_accept_is_serialized_on_pending(store):
    ticket_id = make_ticket(store)

    resolved = store.resolve_handoff(ticket_id, "accept", resolver="human-01")
    assert resolved["status"] == "accepted"
    assert resolved["resolver"] == "human-01"
    assert resolved["resolved_at"]

    # 重复裁决：条件更新零行受影响，返回 None（API 层转为 409）
    assert store.resolve_handoff(ticket_id, "reject") is None


def test_handoff_reject_closes_ticket(store):
    ticket_id = make_ticket(store)
    resolved = store.resolve_handoff(ticket_id, "reject", note="误报")
    assert resolved["status"] == "rejected"


def test_handoff_revise_keeps_pending_and_counts(store):
    ticket_id = make_ticket(store)

    resolved = store.resolve_handoff(ticket_id, "revise", note="请补充卡号后四位")
    assert resolved["status"] == "pending"
    assert resolved["revision_count"] == 1
    assert resolved["revision_note"] == "请补充卡号后四位"

    resolved = store.resolve_handoff(ticket_id, "accept")
    assert resolved["status"] == "accepted"
    assert resolved["revision_count"] == 1  # accept 不改变修订计数


def test_handoff_creation_writes_audit_in_same_tx(store):
    ticket_id = make_ticket(store)
    entries = store.list_audit(resource_type="handoff_ticket", resource_id=ticket_id)
    assert [e["action"] for e in entries] == ["handoff.created"]

    store.resolve_handoff(ticket_id, "accept")
    entries = store.list_audit(resource_type="handoff_ticket", resource_id=ticket_id)
    assert [e["action"] for e in entries] == ["handoff.accepted", "handoff.created"]


def test_handoff_list_filters_by_status(store):
    accepted_id = make_ticket(store)
    store.resolve_handoff(accepted_id, "accept")
    make_ticket(store)

    assert len(store.list_handoffs(status="pending")) == 1
    assert len(store.list_handoffs(status="accepted")) == 1
    assert len(store.list_handoffs(status=None)) == 2


def test_latest_pending_revision_returns_only_revised_pending_ticket(store):
    # 无修订时为空
    assert store.latest_pending_revision("conv-1") is None

    revised_id = make_ticket(store)
    store.resolve_handoff(revised_id, "revise", note="先补充卡号后四位")

    revised = store.latest_pending_revision("conv-1")
    assert revised["ticket_id"] == revised_id
    assert revised["revision_note"] == "先补充卡号后四位"
    assert revised["revision_count"] == 1

    # accept 之后修订上下文不再注入后续运行
    store.resolve_handoff(revised_id, "accept")
    assert store.latest_pending_revision("conv-1") is None


# ── PII 脱敏 ──────────────────────────────────────────────────────────────────

def test_mask_pii_covers_common_identifiers():
    masked = mask_pii("手机 13812345678，身份证 110101199003078515，卡号 6222 0202 0202 0202，邮箱 a.b@example.com")
    assert "13812345678" not in masked
    assert "110101199003078515" not in masked
    assert "6222 0202 0202 0202" not in masked
    assert "a.b@example.com" not in masked
    assert masked.endswith("***@***")
    # 保留首尾便于坐席核对：身份证首 4 尾 4，银行卡仅尾 4
    assert "1101" in masked and "8515" in masked
    assert "6222" not in masked and "0202" in masked


def test_mask_pii_leaves_plain_text_untouched():
    text = "我的账单金额是 1234.56 元，账单日是每月 5 号"
    assert mask_pii(text) == text
    assert mask_pii("") == ""


def test_store_masks_message_and_event_content(store):
    store.create_run("run-1", "conv-1", "u1")
    store.append_message("conv-1", "run-1", "user", "我的手机号是 13812345678")
    store.append_event("run-1", "message.user", {"content": "我的手机号是 13812345678"})

    assert "13812345678" not in store.list_messages("conv-1")[0]["content"]
    assert "13812345678" not in store.list_events("run-1")[0]["payload"]["content"]
