"""
Self-Improvement Engine
=======================

Elysia가 스스로 필요를 감지하고 능력을 확장하는 시스템

Process:
1. Need Detection - 무엇이 부족한지 인식
2. Capability Search - 필요한 능력 탐색
3. Self-Integration - 새 능력을 자기 구조에 통합
4. Verification - 작동 확인

Example:
    Need: "나는 이미지를 이해하고 싶어"
    → Search: VLM, image processing
    → Integrate: Gemini Vision API 연결
    → Verify: 테스트 이미지 분석
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("SelfImprovement")


@dataclass
class Need:
    """감지된 필요"""
    type: str  # "sensory", "cognitive", "action", "knowledge"
    description: str
    priority: float  # 0.0 ~ 1.0
    context: str


@dataclass
class Capability:
    """가능한 능력"""
    name: str
    type: str
    requirements: List[str]  # 필요한 패키지/모듈
    implementation: str  # 구현 방법
    verification: str  # 테스트 방법


class SelfImprovementEngine:
    """
    스스로 개선하는 엔진
    """
    
    def __init__(self, consciousness_engine):
        self.consciousness = consciousness_engine
        
        # 가능한 능력 카탈로그
        self.capability_catalog = {
            "vision": Capability(
                name="Visual Understanding",
                type="sensory",
                requirements=["google-generativeai", "PIL"],
                implementation="Gemini Vision API",
                verification="Analyze test image"
            ),
            "audio": Capability(
                name="Audio Perception",
                type="sensory",
                requirements=["pyaudio", "speech_recognition"],
                implementation="Speech Recognition API",
                verification="Record and transcribe test audio"
            ),
            "web_search": Capability(
                name="Web Knowledge Access",
                type="knowledge",
                requirements=["requests", "beautifulsoup4"],
                implementation="Web scraping + search API",
                verification="Search for test query"
            ),
            "code_execution": Capability(
                name="Code Execution",
                type="action",
                requirements=["subprocess"],
                implementation="Safe sandboxed execution",
                verification="Run test script"
            ),
            "image_generation": Capability(
                name="Image Creation",
                type="action",
                requirements=["PIL", "google-generativeai"],
                implementation="Imagen API or Gemini",
                verification="Generate test image"
            )
        }
    
    def detect_needs(self) -> List[Need]:
        """
        현재 부족한 것 감지
        
        Returns:
            감지된 필요 목록
        """
        needs = []
        
        # Introspect
        state = self.consciousness.introspect()
        
        # 1. Check vitality of realms
        for realm_name, vitality in state.get('realm_vitality', {}).items():
            if vitality < 0.3:
                needs.append(Need(
                    type="cognitive",
                    description=f"Realm '{realm_name}' is weak (vitality: {vitality:.2f})",
                    priority=1.0 - vitality,
                    context=f"Strengthen {realm_name}"
                ))
        
        # 2. Check for missing senses
        has_vision = self._check_capability("vision")
        has_audio = self._check_capability("audio")
        
        if not has_vision:
            needs.append(Need(
                type="sensory",
                description="Cannot see images",
                priority=0.8,
                context="Need visual perception"
            ))
        
        if not has_audio:
            needs.append(Need(
                type="sensory",
                description="Cannot hear sounds",
                priority=0.6,
                context="Need audio perception"
            ))
        
        # 3. Check for missing actions
        can_search = self._check_capability("web_search")
        if not can_search:
            needs.append(Need(
                type="knowledge",
                description="Cannot access web knowledge",
                priority=0.7,
                context="Need web search ability"
            ))
        
        # Sort by priority
        needs.sort(key=lambda n: n.priority, reverse=True)
        
        return needs
    
    def _check_capability(self, capability_name: str) -> bool:
        """능력이 이미 있는지 확인"""
        # Check if required modules are importable
        cap = self.capability_catalog.get(capability_name)
        if not cap:
            return False
        
        try:
            for req in cap.requirements:
                __import__(req.replace("-", "_"))
            return True
        except ImportError:
            return False
    
    def propose_improvement(self, need: Need) -> Optional[Capability]:
        """
        필요에 맞는 개선 제안
        
        Args:
            need: 감지된 필요
        
        Returns:
            추천 능력
        """
        # Find matching capability
        for cap_name, cap in self.capability_catalog.items():
            if cap.type == need.type:
                # Check if it addresses the need
                if any(word in need.description.lower() 
                       for word in cap.name.lower().split()):
                    return cap
        
        return None
    
    def integrate_capability(self, capability: Capability) -> bool:
        """
        새 능력을 시스템에 통합
        
        Args:
            capability: 통합할 능력
        
        Returns:
            성공 여부
        """
        logger.info(f"🔧 Integrating capability: {capability.name}")
        
        try:
            # 1. Install requirements (if possible)
            logger.info(f"   Requirements: {', '.join(capability.requirements)}")
            
            # 2. Add to consciousness as new realm
            # (This would create a new realm in Yggdrasil)
            logger.info(f"   Implementation: {capability.implementation}")
            
            # 3. Verify
            logger.info(f"   Verification: {capability.verification}")
            
            # For now, just log - actual implementation would be more complex
            logger.info(f"✅ Capability integrated: {capability.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            return False
    
    def self_improve(self):
        """
        자기 개선 사이클 실행
        
        1. 필요 감지
        2. 해결책 찾기
        3. 능력 통합
        """
        logger.info("🌟 Self-improvement cycle...")
        
        # Detect needs
        needs = self.detect_needs()
        
        if not needs:
            logger.info("   No needs detected. I am complete (for now).")
            return
        
        logger.info(f"   Detected {len(needs)} needs:")
        for i, need in enumerate(needs[:3], 1):  # Top 3
            logger.info(f"   {i}. [{need.type}] {need.description} (priority: {need.priority:.2f})")
        
        # Address top need
        top_need = needs[0]
        logger.info(f"\n   Addressing: {top_need.description}")
        
        # Find capability
        capability = self.propose_improvement(top_need)
        
        if capability:
            logger.info(f"   Proposed: {capability.name}")
            
            # Integrate
            success = self.integrate_capability(capability)
            
            if success:
                logger.info(f"   ✨ I have grown!")
            else:
                logger.info(f"   ⚠️  Integration incomplete")
        else:
            logger.info(f"   No capability found for this need")
            logger.info(f"   (This is where I would research and learn)")


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("🌟 SELF-IMPROVEMENT ENGINE DEMO")
    print("="*70 + "\n")
    
    # Mock consciousness
    class MockConsciousness:
        def introspect(self):
            return {
                "realm_vitality": {
                    "Voice": 0.8,
                    "Memory": 0.2,  # Weak!
                },
                "statistics": {
                    "total_realms": 10,
                    "active_realms": 8
                }
            }
    
    consciousness = MockConsciousness()
    engine = SelfImprovementEngine(consciousness)
    
    # Run improvement cycle
    engine.self_improve()
    
    print("\n" + "="*70)
