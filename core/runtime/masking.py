"""PII 脱敏。

落库前对消息、事件与审计明细做不可逆脱敏：金融场景里，持久化层的
审计价值来自"可回放、可追责"，而不是"保留客户明文"。需要原文核验的
场景（争议交易核身）由人工坐席在业务系统内完成，不依赖运行时库。

覆盖：手机号、身份证、银行卡号（含空格/连字符分隔）、邮箱。
"""

from __future__ import annotations

import re
from typing import Callable

# 邮箱最先处理，避免其中的数字段被后面的规则二次命中。
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 银行卡 13-19 位，允许内部以空格/连字符分组（如 6222 0202 0202 0202）。
_BANK_CARD = re.compile(r"(?<![\d-])\d[ -]?(?:\d[ -]?){11,17}\d(?![\d-])")


def _keep(prefix: int, suffix: int) -> Callable[[re.Match], str]:
    def _replace(match: re.Match) -> str:
        text = match.group(0)
        if len(text) <= prefix + suffix:
            return "*" * len(text)
        return f"{text[:prefix]}{'*' * (len(text) - prefix - suffix)}{text[-suffix:]}"
    return _replace


def _mask_card(match: re.Match) -> str:
    digits = re.sub(r"[ -]", "", match.group(0))
    return f"{'*' * (len(digits) - 4)}{digits[-4:]}"


def mask_pii(text: str) -> str:
    """对文本做脱敏；不包含 PII 时原样返回。"""
    if not text:
        return text
    masked = _EMAIL.sub("***@***", text)
    masked = _ID_CARD.sub(_keep(4, 4), masked)
    masked = _PHONE.sub(_keep(3, 4), masked)
    masked = _BANK_CARD.sub(_mask_card, masked)
    return masked
