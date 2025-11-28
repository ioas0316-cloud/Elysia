"""
Core/Extensions - 확장 모듈 (낮은 우선순위)
==========================================

낮은 우선순위 개선사항:
1. 모바일 클라이언트 SDK
2. VR/AR 통합
3. 블록체인 기록
"""

from .mobile_sdk import MobileSDK, MobileConfig
from .vr_integration import VRIntegration, VRConfig
from .blockchain_logger import BlockchainLogger, DecisionRecord

__all__ = [
    # Mobile
    "MobileSDK",
    "MobileConfig",
    # VR/AR
    "VRIntegration", 
    "VRConfig",
    # Blockchain
    "BlockchainLogger",
    "DecisionRecord",
]
