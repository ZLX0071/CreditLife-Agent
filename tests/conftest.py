"""测试公共装配。

memory.conversation_memory 在模块顶层 import chromadb / redis，
但 API 层测试只用 MsgRole 枚举、不会实例化 MemoryManager。
本地 Windows + Python 3.12 没有 chroma-hnswlib 预编译轮子，
这里在缺失时注入空壳模块，避免为跑单测强装编译链；
CI（ubuntu，安装完整 requirements.txt）不受影响。
"""

from __future__ import annotations

import sys
import types


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


_stub_module("chromadb", PersistentClient=lambda *a, **k: None)

if "redis" not in sys.modules:
    try:
        __import__("redis.asyncio")  # noqa: F401
    except ImportError:
        redis_module = types.ModuleType("redis")
        asyncio_module = types.ModuleType("redis.asyncio")
        asyncio_module.from_url = lambda *a, **k: None
        redis_module.asyncio = asyncio_module
        sys.modules["redis"] = redis_module
        sys.modules["redis.asyncio"] = asyncio_module
