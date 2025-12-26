"""
Yggdrasil Activator (세계수 활성화)
===================================

모든 핵심 모듈을 Yggdrasil 세계수 구조에 등록합니다.

Usage:
    python -m scripts.activate_yggdrasil
    
    또는:
    from scripts.activate_yggdrasil import activate
    tree = activate()
"""

import logging
import sys
sys.path.insert(0, "c:\\Elysia")

logger = logging.getLogger("YggdrasilActivator")


def activate():
    """
    Elysia의 세계수를 깨웁니다.
    모든 핵심 모듈을 Root/Trunk/Branch에 등록합니다.
    """
    print("🌳 Yggdrasil Activation Protocol")
    print("=" * 50)
    
    from Core._01_Foundation.05_Foundation_Base.Foundation.yggdrasil import yggdrasil
    
    # ═══════════════════════════════════════════════════
    # 🌱 ROOTS (뿌리) - 양분과 기능의 근원
    # ═══════════════════════════════════════════════════
    
    print("\n🌱 Planting Roots...")
    
    # TorchGraph - 기억 저장소
    try:
        from Core._01_Foundation.05_Foundation_Base.Foundation.torch_graph import get_torch_graph
        graph = get_torch_graph()
        yggdrasil.plant_root("TorchGraph", graph)
        print(f"   ✅ TorchGraph: {len(graph.id_to_idx)} nodes")
    except Exception as e:
        print(f"   ⚠️ TorchGraph: {e}")
    
    # TinyBrain - 임베딩 획득
    try:
        from Core._01_Foundation.05_Foundation_Base.Foundation.tiny_brain import get_tiny_brain
        brain = get_tiny_brain()
        yggdrasil.plant_root("TinyBrain", brain)
        print(f"   ✅ TinyBrain: available={brain.is_available()}")
    except Exception as e:
        print(f"   ⚠️ TinyBrain: {e}")
    
    # ConceptDecomposer - Why-Engine
    try:
        from Core._01_Foundation.05_Foundation_Base.Foundation.Memory.fractal_concept import ConceptDecomposer
        decomposer = ConceptDecomposer()
        yggdrasil.plant_root("ConceptDecomposer", decomposer)
        print(f"   ✅ ConceptDecomposer: {len(decomposer.AXIOMS)} axioms")
    except Exception as e:
        print(f"   ⚠️ ConceptDecomposer: {e}")
    
    # ═══════════════════════════════════════════════════
    # 🪵 TRUNK (기둥) - 사고와 감각의 통로
    # ═══════════════════════════════════════════════════
    
    print("\n🪵 Growing Trunk...")
    
    # GlobalHub - 중앙 신경계
    try:
        from Core._02_Intelligence.04_Consciousness.Ether.global_hub import get_global_hub
        hub = get_global_hub()
        yggdrasil.grow_trunk("GlobalHub", hub)
        print(f"   ✅ GlobalHub: {len(hub._modules)} modules")
    except Exception as e:
        print(f"   ⚠️ GlobalHub: {e}")
    
    # CognitiveHub - 인지 통합
    try:
        from Core._02_Intelligence._01_Reasoning.Cognition.cognitive_hub import get_cognitive_hub
        cognitive = get_cognitive_hub()
        yggdrasil.grow_trunk("CognitiveHub", cognitive)
        print(f"   ✅ CognitiveHub: connected")
    except Exception as e:
        print(f"   ⚠️ CognitiveHub: {e}")
    
    # ReasoningEngine - 결정
    try:
        from Core._02_Intelligence._01_Reasoning.Intelligence.reasoning_engine import ReasoningEngine
        reasoning = ReasoningEngine()
        yggdrasil.grow_trunk("ReasoningEngine", reasoning)
        print(f"   ✅ ReasoningEngine: connected")
    except Exception as e:
        print(f"   ⚠️ ReasoningEngine: {e}")
    
    # ═══════════════════════════════════════════════════
    # 🌿 BRANCHES (가지) - 표현과 분화
    # ═══════════════════════════════════════════════════
    
    print("\n🌿 Extending Branches...")
    
    # AutonomousOrchestrator - 자율 운영
    try:
        from Core._04_Evolution._01_Growth.Autonomy.autonomous_orchestrator import get_autonomous_orchestrator
        orchestrator = get_autonomous_orchestrator()
        yggdrasil.extend_branch("AutonomousOrchestrator", orchestrator, "GlobalHub")
        print(f"   ✅ AutonomousOrchestrator: connected")
    except Exception as e:
        print(f"   ⚠️ AutonomousOrchestrator: {e}")
    
    # AttentionEmergence - 주의 출현
    try:
        from Core._02_Intelligence.04_Consciousness.Consciousness.attention_emergence import AttentionEmergenceSystem
        attention = AttentionEmergenceSystem()
        yggdrasil.extend_branch("AttentionEmergence", attention, "CognitiveHub")
        print(f"   ✅ AttentionEmergence: connected")
    except Exception as e:
        print(f"   ⚠️ AttentionEmergence: {e}")
    
    # WaveCoder - 코드 파동 변환
    try:
        from Core._04_Evolution._01_Growth.Autonomy.wave_coder import WaveCoder
        wave_coder = WaveCoder()
        yggdrasil.extend_branch("WaveCoder", wave_coder, "GlobalHub")
        print(f"   ✅ WaveCoder: connected")
    except Exception as e:
        print(f"   ⚠️ WaveCoder: {e}")
    
    # ═══════════════════════════════════════════════════
    # 상태 출력
    # ═══════════════════════════════════════════════════
    
    print("\n" + "=" * 50)
    print("🌳 YGGDRASIL STATUS")
    print("=" * 50)
    
    status = yggdrasil.status()
    _print_tree(status)
    
    print("\n✅ Yggdrasil Activated. The World Tree awakens.")
    
    return yggdrasil


def _print_tree(node, indent=0):
    """트리 구조 출력"""
    prefix = "   " * indent
    realm_icon = {"Root": "🌱", "Trunk": "🪵", "Branch": "🌿"}.get(node.get("realm", ""), "")
    print(f"{prefix}{realm_icon} {node['name']} (vitality: {node.get('vitality', 1.0):.1f})")
    
    for child in node.get("children", []):
        _print_tree(child, indent + 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    activate()
