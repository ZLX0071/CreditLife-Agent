"""SQLite 运行时持久层。

设计原则（对齐生产级 Agent 平台的通用做法）：
  1. 内存不承载真相 —— 会话、运行、事件、工单、审计全部落库，进程重启可回放。
  2. 条件更新保证幂等 —— 终态与人工决策都带 WHERE 前置状态检查，
     重复请求返回冲突而不是重复执行业务。
  3. 审计与业务同事务 —— 工单创建/裁决和 audit_logs 在同一个事务内提交，
     不存在"有工单没审计"的中间态。
  4. 会话级 turn lock —— token + TTL，同一会话同一时刻只允许一个运行；
     进程崩溃后锁随 TTL 自动失效，不需要人工清理。

SQLite 是单文件部署的演示口径；接口按可替换设计，切换 PostgreSQL 时
只需替换 _tx/条件更新的方言，调用方不变。
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

from core.runtime.masking import mask_pii

RUN_ACTIVE = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_CANCELLED = "cancelled"

HANDOFF_PENDING = "pending"
HANDOFF_ACCEPTED = "accepted"
HANDOFF_REJECTED = "rejected"

_TICKET_DECISIONS = {
    "accept": HANDOFF_ACCEPTED,
    "reject": HANDOFF_REJECTED,
    "revise": None,  # revise 保持 pending，成果保留，修订意见随事件回写原运行
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conv_id    TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    TEXT NOT NULL,
    run_id     TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    conv_id      TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    request_id   TEXT NOT NULL DEFAULT '',
    intent       TEXT NOT NULL DEFAULT '',
    agent_type   TEXT NOT NULL DEFAULT '',
    escalated    INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_conv ON runs(conv_id, created_at);
CREATE TABLE IF NOT EXISTS run_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON run_events(run_id, id);
CREATE TABLE IF NOT EXISTS turn_locks (
    conv_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    token       TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS handoff_tickets (
    ticket_id      TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    conv_id        TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    kind           TEXT NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 3,
    summary        TEXT NOT NULL,
    draft          TEXT NOT NULL DEFAULT '{}',
    status         TEXT NOT NULL,
    revision_note  TEXT NOT NULL DEFAULT '',
    revision_count INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    resolved_at    TEXT,
    resolver       TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON handoff_tickets(status, created_at);
CREATE TABLE IF NOT EXISTS audit_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    user_id       TEXT NOT NULL DEFAULT '',
    conv_id       TEXT NOT NULL DEFAULT '',
    run_id        TEXT NOT NULL DEFAULT '',
    action        TEXT NOT NULL,
    resource_type TEXT NOT NULL DEFAULT '',
    resource_id   TEXT NOT NULL DEFAULT '',
    details       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RuntimeStore:
    """线程安全的 SQLite 存储。所有公开方法可在 async 处理函数中直接调用
    （单次写入为微秒级本地事务）；切换远程数据库时改为 to_thread 包装。"""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """显式事务：业务行与审计行要么同时存在，要么都不存在。"""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # ── 会话与消息 ────────────────────────────────────────────────────────────

    def append_message(self, conv_id: str, run_id: str, role: str, content: str) -> None:
        now = _now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO conversations(conv_id, user_id, created_at, updated_at) "
                "VALUES(?, '', ?, ?) "
                "ON CONFLICT(conv_id) DO UPDATE SET updated_at = excluded.updated_at",
                (conv_id, now, now),
            )
            conn.execute(
                "INSERT INTO messages(conv_id, run_id, role, content, created_at) VALUES(?,?,?,?,?)",
                (conv_id, run_id, role, mask_pii(content), now),
            )

    def list_messages(self, conv_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, conv_id, run_id, role, content, created_at FROM messages "
            "WHERE conv_id = ? ORDER BY id DESC LIMIT ?",
            (conv_id, max(1, min(limit, 500))),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    # ── 运行 ─────────────────────────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        conv_id: str,
        user_id: str,
        request_id: str = "",
        intent: str = "",
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs(run_id, conv_id, user_id, request_id, intent, status, created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, conv_id, user_id, request_id, intent, RUN_ACTIVE, _now()),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        escalated: Optional[bool] = None,
        agent_type: Optional[str] = None,
    ) -> bool:
        """条件终态更新：只有仍处于 running 的运行可被终结。

        返回 False 表示运行已终结（重复回调/对账竞争），调用方必须当作幂等
        no-op 处理，而不是重复写终态事件。
        """
        assert status in (RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED)
        sets = ["status = ?", "completed_at = ?"]
        params: List[Any] = [status, _now()]
        if escalated is not None:
            sets.append("escalated = ?")
            params.append(int(escalated))
        if agent_type is not None:
            sets.append("agent_type = ?")
            params.append(agent_type)
        params.extend([run_id, RUN_ACTIVE])
        with self._tx() as conn:
            cursor = conn.execute(
                f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ? AND status = ?",
                params,
            )
            return cursor.rowcount > 0

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(self, conv_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE conv_id = ? ORDER BY created_at DESC LIMIT ?",
            (conv_id, max(1, min(limit, 200))),
        ).fetchall()
        return [dict(row) for row in rows]

    def cancel_run(self, run_id: str, *, reason: str = "", actor: str = "agent-console") -> bool:
        """人工/运维取消一次运行：条件更新 running → cancelled，审计同事务。

        与 finish_run 同一条规则：False 表示运行已终结（或不存在的对账竞争），
        调用方必须当作幂等 no-op 处理。
        """
        now = _now()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT conv_id, user_id FROM runs WHERE run_id = ? AND status = ?",
                (run_id, RUN_ACTIVE),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ? AND status = ?",
                (RUN_CANCELLED, now, run_id, RUN_ACTIVE),
            )
            conn.execute(
                "INSERT INTO audit_logs(ts, user_id, conv_id, run_id, action, resource_type, resource_id, details) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (now, row["user_id"], row["conv_id"], run_id, "run.cancelled", "run", run_id,
                 mask_pii(json.dumps({"reason": reason, "actor": actor}, ensure_ascii=False))),
            )
        return True

    def reconcile_stale_runs(self, max_age_s: float = 600.0) -> List[Dict[str, Any]]:
        """僵尸运行对账：进程在 create_run 与 finish_run 之间崩溃会留下永远
        running 的记录。启动时把超过 max_age 仍处 running 的运行归档为
        cancelled（cancel_run 的同事务审计同样生效），返回被归档的运行。"""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_s)).isoformat(
            timespec="milliseconds"
        )
        rows = self._conn.execute(
            "SELECT run_id, conv_id, created_at FROM runs WHERE status = ? AND created_at < ?",
            (RUN_ACTIVE, cutoff),
        ).fetchall()
        reconciled = []
        for row in rows:
            if self.cancel_run(row["run_id"], reason="reconcile_stale_on_startup", actor="system"):
                reconciled.append(dict(row))
        return reconciled

    # ── 运行事件（断线恢复的回放源）──────────────────────────────────────────

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO run_events(run_id, type, payload, created_at) VALUES(?,?,?,?)",
                (run_id, event_type, mask_pii(json.dumps(payload, ensure_ascii=False)), _now()),
            )
            return int(cursor.lastrowid)

    def list_events(self, run_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
        """按 seq 回放事件；断线客户端带上已消费的 seq 即可增量续读。"""
        rows = self._conn.execute(
            "SELECT id AS seq, run_id, type, payload, created_at FROM run_events "
            "WHERE run_id = ? AND id > ? ORDER BY id",
            (run_id, max(0, after_seq)),
        ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event["payload"])
            events.append(event)
        return events

    def latest_pending_revision(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """取该会话最近一条带修订意见的 pending 工单，供下一次运行注入修订上下文。

        revise 不终结工单，意见只写在工单上——这里是把"坐席要求"重新送进
        对话闭环的唯一读取口，保证修订不会被静默丢掉。
        """
        row = self._conn.execute(
            "SELECT ticket_id, run_id, kind, revision_note, revision_count, created_at "
            "FROM handoff_tickets WHERE conv_id = ? AND status = ? AND revision_count > 0 "
            "ORDER BY created_at DESC LIMIT 1",
            (conv_id, HANDOFF_PENDING),
        ).fetchone()
        return dict(row) if row else None

    # ── 会话级 turn lock ─────────────────────────────────────────────────────

    def acquire_turn_lock(self, conv_id: str, run_id: str, ttl_s: float = 180.0) -> Optional[str]:
        """获取会话执行锁。

        返回 token 表示获取成功（同一 run_id 重入返回原 token，对应重试场景）；
        返回 None 表示锁被其它运行持有且未过期，调用方应返回 409。
        锁不主动释放时随 TTL 自动失效，进程崩溃不会造成死锁。
        """
        now = datetime.now(timezone.utc)
        existing = self._conn.execute(
            "SELECT run_id, token, expires_at FROM turn_locks WHERE conv_id = ?",
            (conv_id,),
        ).fetchone()
        if existing is not None:
            if existing["run_id"] == run_id and existing["expires_at"] > now.isoformat(timespec="milliseconds"):
                return existing["token"]
            if existing["expires_at"] > now.isoformat(timespec="milliseconds"):
                return None
        token = secrets.token_hex(16)
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO turn_locks(conv_id, run_id, token, acquired_at, expires_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(conv_id) DO UPDATE SET "
                "run_id = excluded.run_id, token = excluded.token, "
                "acquired_at = excluded.acquired_at, expires_at = excluded.expires_at "
                "WHERE turn_locks.expires_at <= excluded.acquired_at OR turn_locks.run_id = excluded.run_id",
                (conv_id, run_id, token, now.isoformat(timespec="milliseconds"),
                 (now + timedelta(seconds=ttl_s)).isoformat(timespec="milliseconds")),
            )
            if cursor.rowcount == 0:
                return None
        return token

    def heartbeat_turn_lock(self, conv_id: str, token: str, ttl_s: float = 180.0) -> bool:
        """长调用续租；token 不匹配说明锁已易主，调用方必须停止本次执行。"""
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_s)
        ).isoformat(timespec="milliseconds")
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE turn_locks SET expires_at = ? WHERE conv_id = ? AND token = ?",
                (expires_at, conv_id, token),
            )
            return cursor.rowcount > 0

    def release_turn_lock(self, conv_id: str, token: str) -> bool:
        with self._tx() as conn:
            cursor = conn.execute(
                "DELETE FROM turn_locks WHERE conv_id = ? AND token = ?", (conv_id, token)
            )
            return cursor.rowcount > 0

    def active_lock(self, conv_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT conv_id, run_id, acquired_at, expires_at FROM turn_locks WHERE conv_id = ?",
            (conv_id,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= datetime.now(timezone.utc).isoformat(timespec="milliseconds"):
            return None
        return dict(row)

    # ── 人工确认门（升级工单）────────────────────────────────────────────────

    def create_handoff(
        self,
        *,
        run_id: str,
        conv_id: str,
        user_id: str,
        kind: str,
        summary: str,
        draft: Optional[Dict[str, Any]] = None,
        priority: int = 3,
    ) -> str:
        """创建待人工确认的处置工单。AI 的升级建议默认不自动执行，
        必须经过 accept / reject / revise 的显式人工决策。"""
        ticket_id = str(uuid.uuid4())[:12]
        now = _now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO handoff_tickets(ticket_id, run_id, conv_id, user_id, kind, priority, "
                "summary, draft, status, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ticket_id, run_id, conv_id, user_id, kind, priority,
                 mask_pii(summary), json.dumps(draft or {}, ensure_ascii=False), HANDOFF_PENDING, now),
            )
            conn.execute(
                "INSERT INTO audit_logs(ts, user_id, conv_id, run_id, action, resource_type, resource_id, details) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (now, user_id, conv_id, run_id, "handoff.created", "handoff_ticket", ticket_id,
                 json.dumps({"kind": kind, "priority": priority}, ensure_ascii=False)),
            )
        return ticket_id

    def get_handoff(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            return None
        ticket = dict(row)
        ticket["draft"] = json.loads(ticket["draft"])
        return ticket

    def list_handoffs(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM handoff_tickets WHERE status = ? ORDER BY priority, created_at LIMIT ?",
                (status, max(1, min(limit, 200))),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM handoff_tickets ORDER BY priority, created_at LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        tickets = []
        for row in rows:
            ticket = dict(row)
            ticket["draft"] = json.loads(ticket["draft"])
            tickets.append(ticket)
        return tickets

    def resolve_handoff(
        self,
        ticket_id: str,
        decision: str,
        note: str = "",
        resolver: str = "agent-console",
    ) -> Optional[Dict[str, Any]]:
        """人工裁决工单。决策串行在 pending 状态上（条件更新），并发重复
        提交只有一个生效，其余拿到 None——由路由层转成 409。

        - accept / reject：写入终态与裁决人，审计同事务。
        - revise：状态保持 pending，旧建议单原样保留在运行事件里，
          修订意见入库并回写原运行事件流，供下一次续跑参考。
        """
        if decision not in _TICKET_DECISIONS:
            raise ValueError(f"非法决策: {decision}")
        now = _now()
        action = f"handoff.{decision}ed" if decision in ("accept", "reject") else "handoff.revised"
        with self._tx() as conn:
            row = conn.execute(
                "SELECT status, run_id, conv_id, user_id FROM handoff_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if row is None or row["status"] != HANDOFF_PENDING:
                return None
            if decision == "revise":
                conn.execute(
                    "UPDATE handoff_tickets SET revision_note = ?, revision_count = revision_count + 1 "
                    "WHERE ticket_id = ? AND status = ?",
                    (mask_pii(note), ticket_id, HANDOFF_PENDING),
                )
            else:
                conn.execute(
                    "UPDATE handoff_tickets SET status = ?, resolved_at = ?, resolver = ? "
                    "WHERE ticket_id = ? AND status = ?",
                    (_TICKET_DECISIONS[decision], now, resolver, ticket_id, HANDOFF_PENDING),
                )
            conn.execute(
                "INSERT INTO audit_logs(ts, user_id, conv_id, run_id, action, resource_type, resource_id, details) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (now, row["user_id"], row["conv_id"], row["run_id"], action,
                 "handoff_ticket", ticket_id,
                 mask_pii(json.dumps({"note": note, "resolver": resolver}, ensure_ascii=False))),
            )
        return self.get_handoff(ticket_id)

    # ── 审计 ─────────────────────────────────────────────────────────────────

    def audit(
        self,
        *,
        action: str,
        user_id: str = "",
        conv_id: str = "",
        run_id: str = "",
        resource_type: str = "",
        resource_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO audit_logs(ts, user_id, conv_id, run_id, action, resource_type, resource_id, details) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (_now(), user_id, conv_id, run_id, action, resource_type, resource_id,
                 mask_pii(json.dumps(details or {}, ensure_ascii=False))),
            )

    def list_audit(
        self,
        action: str = "",
        resource_type: str = "",
        resource_id: str = "",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM audit_logs"
        conditions: List[str] = []
        params: List[Any] = []
        if action:
            conditions.append("action = ?")
            params.append(action)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if resource_id:
            conditions.append("resource_id = ?")
            params.append(resource_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        rows = self._conn.execute(query, params).fetchall()
        entries = []
        for row in rows:
            entry = dict(row)
            entry["details"] = json.loads(entry["details"])
            entries.append(entry)
        return entries
