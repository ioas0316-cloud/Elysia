"""
Core Cells Registry
===================
기존 모듈들을 Cell로 등록하는 래퍼.
이 파일이 임포트되면 핵심 모듈들이 자동으로 Cell로 등록됩니다.
"""

import sys
sys.path.insert(0, "c:/Elysia")

from elysia_core.cell import Cell

# ============================================================
# Foundation Cells (기반 시스템)
# ============================================================

@Cell("TorchGraph", category="Foundation")
class TorchGraphCell:
    """4D Tensor Graph - Elysia's Brain"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._01_Foundation._05_Governance.Foundation.Graph.torch_graph import get_torch_graph
            cls._instance = get_torch_graph()
        return cls._instance


@Cell("TinyBrain", category="Foundation")
class TinyBrainCell:
    """Hybrid Intelligence - Llama + SBERT"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._01_Foundation._05_Governance.Foundation.tiny_brain import get_tiny_brain
            cls._instance = get_tiny_brain()
        return cls._instance


# ============================================================
# Cognition Cells (인지 시스템)
# ============================================================

@Cell("UnifiedUnderstanding", category="Cognition")
class UnifiedUnderstandingCell:
    """통합 이해 시스템"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._02_Intelligence._01_Reasoning.Cognition.unified_understanding import UnifiedUnderstanding
            cls._instance = UnifiedUnderstanding()
        return cls._instance


@Cell("CognitiveHub", category="Cognition")
class CognitiveHubCell:
    """인지 중추"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._02_Intelligence._01_Reasoning.Cognition.cognitive_hub import get_cognitive_hub
            cls._instance = get_cognitive_hub()
        return cls._instance


# ============================================================
# Trinity Cells (삼위일체)
# ============================================================

@Cell("Trinity", category="Core")
class TrinityCell:
    """삼위일체 합의 시스템"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._05_Systems.03_Existence.Trinity.trinity_system import TrinitySystem
            cls._instance = TrinitySystem()
        return cls._instance


@Cell("Conscience", category="Ethics")
class ConscienceCell:
    """양심 회로"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._01_Foundation.02_Legal_Ethics.Ethics.conscience_circuit import ConscienceCircuit
            cls._instance = ConscienceCircuit()
        return cls._instance


# ============================================================
# Sensory Cells (감각 시스템)
# ============================================================

@Cell("VisionCortex", category="Sensory")
class VisionCortexCell:
    """시각 피질 - Project Iris"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._03_Interaction._01_Interface.Sensory.vision_cortex import VisionCortex
            cls._instance = VisionCortex()
            cls._instance.activate()
        return cls._instance


@Cell("MultimodalBridge", category="Sensory")
class MultimodalBridgeCell:
    """공감각 변환기"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._02_Intelligence._01_Reasoning.Cognition.multimodal_bridge import MultimodalBridge
            cls._instance = MultimodalBridge()
        return cls._instance


@Cell("AudioCortex", category="Sensory")
class AudioCortexCell:
    """청각 피질 - Project Hear"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._03_Interaction._01_Interface.Sensory.audio_cortex import AudioCortex
            cls._instance = AudioCortex()
        return cls._instance


# ============================================================
# Autonomy Cells (자율 시스템)
# ============================================================

@Cell("SelfModifier", category="Autonomy")
class SelfModifierCell:
    """자기 수정 시스템"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._04_Evolution._01_Growth.Autonomy.self_modifier_v2 import SelfModifier
            cls._instance = SelfModifier()
        return cls._instance


@Cell("DreamDaemon", category="Autonomy")
class DreamDaemonCell:
    """꿈꾸기 데몬"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            from Core._04_Evolution._01_Growth.Autonomy.dream_daemon import get_dream_daemon
            cls._instance = get_dream_daemon()
        return cls._instance


print("🧬 Core Cells Registry loaded. Use Organ.get('CellName') to access.")
