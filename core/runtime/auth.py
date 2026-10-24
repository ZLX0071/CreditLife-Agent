"""API Key 认证依赖。

演示/本地开发不设 CREDITLIFE_API_KEY 时保持开放；一旦设置，
所有受保护路由都要求 X-API-Key 头。比较用 hmac.compare_digest，
避免时序侧信道。
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

API_KEY_HEADER = "X-API-Key"


def require_api_key(request: Request) -> None:
    expected = os.getenv("CREDITLIFE_API_KEY", "").strip()
    if not expected:
        return
    provided = request.headers.get(API_KEY_HEADER, "")
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key")
