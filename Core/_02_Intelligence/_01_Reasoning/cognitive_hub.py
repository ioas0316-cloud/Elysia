"""
Cognitive Hub (인지 통합 허브)
==============================
"새로 만들지 않는다. 있는 것을 연결한다."

This module INTEGRATES existing cognitive systems:
- PrincipleDistiller (원리 추출)
- ExperienceLearner (경험 학습, 메타학습)
- CausalNarrativeEngine (인과 서사, Why 추적)
- SelfCorrection (자기수정)

NO NEW SYSTEMS. Just connections.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("CognitiveHub")

# ============================================================================
# Import Existing Systems (새로 만들지 않음)
# ============================================================================

try:
    from Core._02_Intelligence._01_Reasoning.Cognition.principle_distiller import get_principle_distiller, PrincipleDistiller
    PRINCIPLE_AVAILABLE = True
except ImportError:
    PRINCIPLE_AVAILABLE = False
    logger.warning("⚠️ PrincipleDistiller not available")

try:
    from Core._01_Foundation._05_Governance.Foundation.experience_learner import ExperienceLearner, Experience
    EXPERIENCE_AVAILABLE = True
except ImportError:
    try:
        # Fallback: 다른 위치 시도
        from experience_learner import ExperienceLearner, Experience
        EXPERIENCE_AVAILABLE = True
    except ImportError:
        EXPERIENCE_AVAILABLE = False
        logger.warning("⚠️ ExperienceLearner not available")

try:
    from Core._01_Foundation._05_Governance.Foundation.causal_narrative_engine import CausalNarrativeEngine
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    logger.warning("⚠️ CausalNarrativeEngine not available")

try:
    from Core._02_Intelligence._01_Reasoning.Intelligence.Logos.philosophical_core import PhilosophicalCore
    PHILOSOPHY_AVAILABLE = True
except ImportError:
    PHILOSOPHY_AVAILABLE = False

try:
    # After Foundation split, torch_graph is in Graph/
    from Core._01_Foundation._05_Governance.Foundation.Graph.torch_graph import get_torch_graph
    GRAPH_AVAILABLE = True
except ImportError:
    try:
        # Fallback to redirect stub
        from Core._01_Foundation._05_Governance.Foundation.torch_graph import get_torch_graph
        GRAPH_AVAILABLE = True
    except ImportError:
        GRAPH_AVAILABLE = False


@dataclass
class CognitiveResult:
    """인지 처리 결과"""
    concept: str
    principle: Dict[str, str]  # From PrincipleDistiller
    why_chain: List[str]  # From CausalNarrativeEngine
    patterns: List[Dict]  # From ExperienceLearner
    corrections: List[Dict]  # Self-corrections applied
    meta_insights: Dict  # From meta-learning


class CognitiveHub:
    """
    인지 통합 허브 (Cognitive Integration Hub)
    
    기존 시스템들을 연결하여 통합된 인지 파이프라인을 제공합니다.
    새로운 기능을 구현하지 않고, 기존 모듈들의 메서드를 조합합니다.
    """
    
    def __init__(self):
        logger.info("🧠 CognitiveHub initializing (connecting existing systems)...")
        
        # Connect existing systems (새로 만들지 않음)
        self.principle_distiller = get_principle_distiller() if PRINCIPLE_AVAILABLE else None
        self.experience_learner = ExperienceLearner() if EXPERIENCE_AVAILABLE else None
        self.causal_engine = CausalNarrativeEngine() if CAUSAL_AVAILABLE else None
        self.graph = get_torch_graph() if GRAPH_AVAILABLE else None
        
        # GlobalHub 연결 (Central Nervous System)
        self._hub = None
        try:
            from Core._02_Intelligence.04_Consciousness.Ether.global_hub import get_global_hub
            self._hub = get_global_hub()
            self._hub.register_module(
                "CognitiveHub",
                "Core/Cognition/cognitive_hub.py",
                ["understanding", "analysis", "learning", "why"],
                "The Mind - integrates cognitive systems"
            )
            self._hub.subscribe("CognitiveHub", "concept", self._on_concept_event, weight=0.9)
            self._hub.subscribe("CognitiveHub", "thought", self._on_thought_event, weight=0.8)
            logger.info("   ✅ GlobalHub connected")
        except ImportError:
            logger.warning("   ⚠️ GlobalHub not available")
        
        # Status
        connected = sum([
            PRINCIPLE_AVAILABLE,
            EXPERIENCE_AVAILABLE,
            CAUSAL_AVAILABLE,
            GRAPH_AVAILABLE
        ])
        
        logger.info(f"✅ CognitiveHub ready. {connected}/4 systems connected.")
    
    def _on_concept_event(self, event):
        """React to concept events from GlobalHub."""
        logger.debug(f"   💡 CognitiveHub received concept from {event.source}")
        if event.payload and "concept" in event.payload:
            return self.understand(event.payload["concept"])
        return {"acknowledged": True}
    
    def _on_thought_event(self, event):
        """React to thought events from GlobalHub."""
        logger.debug(f"   💭 CognitiveHub received thought from {event.source}")
        return {"acknowledged": True}
    
    def understand(self, concept: str) -> CognitiveResult:
        """
        개념을 통합적으로 이해합니다.
        
        Pipeline:
        1. PrincipleDistiller.distill() → 원리 추출
        2. CausalNarrativeEngine.explain_why() → 인과 사슬
        3. ExperienceLearner.get_recommendations() → 관련 패턴
        4. Store result in TorchGraph
        """
        logger.info(f"🔍 Understanding: '{concept}'")
        
        result = CognitiveResult(
            concept=concept,
            principle={},
            why_chain=[],
            patterns=[],
            corrections=[],
            meta_insights={}
        )
        
        # 1. Principle Extraction (기존 시스템 사용)
        if self.principle_distiller:
            result.principle = self.principle_distiller.distill(concept)
            logger.debug(f"   Principle: {result.principle.get('principle', 'N/A')}")
        
        # 2. Why Chain (기존 시스템 사용)
        if self.causal_engine:
            # CausalNarrativeEngine.explain_why() 호출
            try:
                why_explanations = self.causal_engine.explain_why(concept)
                result.why_chain = why_explanations if why_explanations else []
            except Exception as e:
                logger.debug(f"   Why-chain: {e}")
        
        # 3. Related Patterns (기존 시스템 사용)
        if self.experience_learner:
            context = {"concept": concept, "type": "understanding"}
            recommendations = self.experience_learner.get_recommendations(context)
            result.patterns = recommendations if recommendations else []
        
        # 4. Store in Graph (기존 시스템 사용)
        if self.graph and result.principle:
            node_id = f"Concept:{concept}"
            self.graph.add_node(
                node_id,
                metadata={
                    "principle": result.principle,
                    "why_chain": result.why_chain,
                    "cognitive_hub": True
                }
            )
        
        return result
    
    def learn_from(self, concept: str, feedback: float, context: Dict = None) -> Dict:
        """
        경험으로부터 학습합니다.
        
        Pipeline:
        1. ExperienceLearner.learn_from_experience() → 패턴 추출
        2. ExperienceLearner.meta_learn() → 학습 전략 개선
        3. Store corrections in CausalNarrativeEngine
        """
        if not self.experience_learner:
            return {"error": "ExperienceLearner not available"}
        
        logger.info(f"📚 Learning from: '{concept}' (feedback={feedback})")
        
        # 1. Create Experience (기존 구조 사용)
        exp = Experience(
            timestamp=__import__('time').time(),
            context=context or {"concept": concept},
            action={"type": "understand", "target": concept},
            outcome={"success": feedback > 0.5},
            feedback=feedback,
            layer="cognitive"
        )
        
        # 2. Learn (기존 메서드 호출)
        learn_result = self.experience_learner.learn_from_experience(exp)
        
        # 3. Meta-learn (기존 메서드 호출)
        meta_result = self.experience_learner.meta_learn()
        
        return {
            "learning": learn_result,
            "meta": meta_result
        }
    
    def ask_why(self, statement: str) -> List[str]:
        """
        "왜?"를 묻습니다.
        
        기존 CausalNarrativeEngine.explain_why() 를 직접 호출합니다.
        """
        if not self.causal_engine:
            return ["CausalNarrativeEngine not available"]
        
        return self.causal_engine.explain_why(statement)
    
    def get_status(self) -> Dict:
        """허브 상태 반환"""
        return {
            "principle_distiller": PRINCIPLE_AVAILABLE,
            "experience_learner": EXPERIENCE_AVAILABLE,
            "causal_engine": CAUSAL_AVAILABLE,
            "graph": GRAPH_AVAILABLE,
            "experience_count": self.experience_learner.experience_count if self.experience_learner else 0
        }


# ============================================================================
# Singleton
# ============================================================================

_hub = None

def get_cognitive_hub() -> CognitiveHub:
    global _hub
    if _hub is None:
        _hub = CognitiveHub()
    return _hub


# ============================================================================
# Demo (기존 시스템 연결 확인용)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("🧠 Cognitive Hub Demo")
    print("=" * 50)
    print("(새로 만들지 않음. 기존 시스템 연결만 수행.)")
    print()
    
    hub = get_cognitive_hub()
    
    print("📊 Connected Systems:")
    status = hub.get_status()
    for system, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {system}: {available}")
    
    print()
    print("✅ CognitiveHub initialized.")
    print("   Use hub.understand('concept') to process through all systems.")
    print("   Use hub.learn_from('concept', feedback) to learn.")
    print("   Use hub.ask_why('statement') to trace causality.")
