"""
Core/API - API 서버 및 외부 연동 모듈
=====================================

중간 우선순위 개선사항:
1. API 서버 (FastAPI REST) - 외부 연동
2. 웹 대시보드 - 실시간 시각화
3. LLM 연동 (LangChain) - 고급 대화
"""

from .server import ElysiaAPI, APIConfig, create_app
from .dashboard import DashboardServer, ConsciousnessState, ResonanceSnapshot
from .llm_bridge import LLMBridge, ConversationContext, LLMConfig

__all__ = [
    # API Server
    "ElysiaAPI",
    "APIConfig", 
    "create_app",
    # Dashboard
    "DashboardServer",
    "ConsciousnessState",
    "ResonanceSnapshot",
    # LLM Bridge
    "LLMBridge",
    "ConversationContext",
    "LLMConfig",
]
