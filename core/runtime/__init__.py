"""core.runtime — 运行时持久层（唯一公开入口）。

对外只暴露 RuntimeStore、mask_pii 与 require_api_key；
禁止绕过本包深导入内部模块。
"""

from core.runtime.auth import API_KEY_HEADER, require_api_key
from core.runtime.masking import mask_pii
from core.runtime.store import (
    RUN_ACTIVE,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    HANDOFF_ACCEPTED,
    HANDOFF_PENDING,
    HANDOFF_REJECTED,
    RuntimeStore,
)

__all__ = [
    "RuntimeStore",
    "mask_pii",
    "require_api_key",
    "API_KEY_HEADER",
    "RUN_ACTIVE",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_CANCELLED",
    "HANDOFF_PENDING",
    "HANDOFF_ACCEPTED",
    "HANDOFF_REJECTED",
]
