"""集中配置常量。

模型名的唯一权威来源：默认值面向官方 Anthropic API，
本地/生产通过 ANTHROPIC_MODEL（及各角色 CREDITLIFE_*_MODEL）
覆盖为兼容端点上的模型串。历史版本曾把同一默认值散落在 6 个文件里。
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

# 危机干预关键词的唯一权威来源：命中即无条件直达人工升级，
# 不经过意图识别的概率投票。路由前置拦截与升级节点共用本列表。
CRISIS_KEYWORDS = ("轻生", "跳楼", "不想活", "活不下去", "自残", "自杀", "逼死")
