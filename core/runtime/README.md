# core/runtime — 运行时持久层

本目录是 CreditLife 的"可审计运行时"：会话、运行、事件、升级工单、审计全部落库。
内存（trace 窗口、监控统计）只承载可丢弃的派生状态，**内存不承载真相**。

## 不变量（修改任何文件前先读完这一节）

1. **单一事实源在 SQLite。** 一次用户消息 = 一条 `runs` 记录 + 一串 `run_events` +
   至少两条 `messages`（用户/助手）+ 至少一条 `chat.turn` 审计。任何"从内存重建"的做法都是错的。
2. **终态只能条件更新。** `finish_run` 只允许 `running → 终态`，重复终结返回 False。
   调用方必须把 False 当作幂等 no-op，不得覆盖已有终态。
3. **人工决策串行在 pending 上。** `resolve_handoff` 用条件更新实现：
   并发/重复裁决只有一个生效，其余拿 None（API 层转 409）。
   revise 保持 pending：原建议单保留，修订意见回写原运行事件流。
4. **审计与业务同事务。** 工单创建/裁决与 audit_logs 在同一个 `BEGIN IMMEDIATE` 事务内提交。
   不允许出现"有工单没审计"的中间态。
5. **落库即脱敏。** 消息、事件、工单摘要、审计明细写入前过 `mask_pii`。
   需要明文核验的场景由人工坐席在业务系统完成，运行时库不存客户明文。
6. **会话级执行锁。** 同一 conv_id 同一时刻最多一个 running；锁带 TTL 自动失效，
   进程崩溃不会死锁。409 响应必须带 `active_run_id`，客户端据此回放 `GET /runs/{id}/events`。
7. **修订意见必须回流。** revise 保持 pending，修订意见除了回写原运行事件流，
   还必须通过 `latest_pending_revision` 注入同会话下一次运行（事件 `handoff.revision_applied`
   + 模型上下文）。工单 accept/reject 后注入即停止——闭环靠读取口保证，不靠人记得。
8. **running 的三个终态必须都可达且可对账。** completed / failed / cancelled 都是
   条件更新（不覆盖已有终态）；取消与启动僵尸对账走同一条 `cancel_run`，审计同事务——
   看板上的 running 永远代表真的在跑。

## 目录

| 文件 | 职责 |
| --- | --- |
| `store.py` | SQLite 存储与全部不变量的实现（唯一写入方） |
| `masking.py` | PII 脱敏规则（手机号/身份证/银行卡/邮箱） |
| `auth.py` | X-API-Key 认证依赖（未配置则本地开放） |

## 公开入口

只允许 `from core.runtime import RuntimeStore, mask_pii, require_api_key`，禁止深导入。

## 验证命令

```bash
pytest tests -q -k "runtime_store or api_runtime"
```

## 已知边界（诚实清单）

- SQLite 单文件适合演示与单机部署；切 PostgreSQL 时替换 `_tx`/条件更新方言，调用方不变。
- turn lock 无后台心跳续租；模型调用超过锁 TTL（默认 180s）的场景需要加心跳，
  当前对话最坏延迟约 25s，留有 7 倍余量。
- 僵尸对账只在启动时执行一次；多实例部署时需要改为带分布式租约的对账，避免误杀其他实例的活跃运行。
- `chat.failed` 审计记录的是异常摘要；完整堆栈在应用日志，不落审计库。
