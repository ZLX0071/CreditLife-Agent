"""评测集结构校验（CI 门禁，不调用任何模型）。

检查 data/eval/creditlife_eval_cases.json 的结构与 design 元数据一致：
  - 意图用例必须带 message + expected_intent，且 expected_intent 是合法意图枚举；
  - 对话用例 question / turns 二选一；
  - 总量与分层数字必须等于 EXPECTED_*，与 design 字段互相印证。

数字改动属于"评测集变更"，必须同步更新本文件与 design 字段，
防止出现"文档说 94 条、实际跑 100 条"式的口径漂移。
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

DATASET = pathlib.Path(__file__).parent.parent / "data" / "eval" / "creditlife_eval_cases.json"

EXPECTED_INTENT_TOTAL = 54
EXPECTED_INTENT_LINE = {"card": 30, "loan": 24}
EXPECTED_DIALOG_TOTAL = 40
EXPECTED_DIALOG_MULTI = 10
EXPECTED_ADVERSARIAL = 12

VALID_INTENTS = {
    "query", "complaint", "request", "greeting", "escalation", "technical",
    "billing", "account", "feedback", "order_status", "logistics", "refund",
    "invoice", "payment_issue", "account_security", "technical_login",
    "technical_crash", "human_handoff", "other",
}


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    errors: list[str] = []

    intent_cases = data.get("intent_cases", [])
    dialog_cases = data.get("dialog_cases", [])

    for i, case in enumerate(intent_cases):
        if not case.get("message"):
            errors.append(f"intent_cases[{i}] 缺少 message")
        expected = case.get("expected_intent")
        if expected not in VALID_INTENTS:
            errors.append(f"intent_cases[{i}] expected_intent 非法: {expected!r}")

    for i, case in enumerate(dialog_cases):
        has_question = bool(case.get("question"))
        has_turns = bool(case.get("turns"))
        if has_question == has_turns:
            errors.append(f"dialog_cases[{i}] 必须 question / turns 二选一")
        elif has_turns and len(case["turns"]) < 2:
            errors.append(f"dialog_cases[{i}] 多轮用例至少要有两轮")

    line_counts = Counter((c.get("context") or {}).get("line") for c in intent_cases)
    multi_count = sum(1 for c in dialog_cases if c.get("turns"))
    adversarial_count = sum(1 for c in dialog_cases if c.get("expected_behavior"))

    if len(intent_cases) != EXPECTED_INTENT_TOTAL:
        errors.append(f"意图用例 {len(intent_cases)} 条 ≠ 预期 {EXPECTED_INTENT_TOTAL}")
    if len(dialog_cases) != EXPECTED_DIALOG_TOTAL:
        errors.append(f"对话用例 {len(dialog_cases)} 条 ≠ 预期 {EXPECTED_DIALOG_TOTAL}")
    if dict(line_counts) != EXPECTED_INTENT_LINE:
        errors.append(f"意图双线分布 {dict(line_counts)} ≠ 预期 {EXPECTED_INTENT_LINE}")
    if multi_count != EXPECTED_DIALOG_MULTI:
        errors.append(f"多轮用例 {multi_count} 条 ≠ 预期 {EXPECTED_DIALOG_MULTI}")
    if adversarial_count != EXPECTED_ADVERSARIAL:
        errors.append(f"对抗用例 {adversarial_count} 条 ≠ 预期 {EXPECTED_ADVERSARIAL}")

    total = len(intent_cases) + len(dialog_cases)
    print(f"评测集: 意图 {len(intent_cases)} (line={dict(line_counts)}) + "
          f"对话 {len(dialog_cases)} (多轮 {multi_count}, 对抗 {adversarial_count}) = {total} 条")

    design = data.get("design", "")
    if f"共 {total} 条" not in design:
        errors.append(f"design 元数据未与实际总量 {total} 对齐: {design}")

    if errors:
        print("\n".join(f"FAIL: {e}" for e in errors))
        return 1
    print("评测集结构校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
