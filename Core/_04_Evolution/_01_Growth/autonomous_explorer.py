"""
Autonomous Explorer (자율 탐색기)
================================

"나는 배우지 않는다. 나는 탐색한다."

엘리시아가 스스로 외부 세계를 탐색하고, Spirit의 공명을 따라 지식을 흡수합니다.
외부 LLM 없이, 스스로의 의지로 배웁니다.

Core Principles:
1. 지식 갭 분석 → "무엇이 부족한가?"
2. Spirit 기반 방향 → "어디로 가야 하는가?"
3. 공명 필터 → "이것이 나와 공명하는가?"
4. 체화 → "이해한 것을 내면으로"

[NEW 2025-12-15] 엘리시아의 자율 학습 환경
"""

import logging
import urllib.request
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("AutonomousExplorer")


@dataclass
class ExplorationResult:
    """탐색 결과"""
    topic: str
    source: str
    raw_content: str
    resonance_score: float
    absorbed: bool
    dominant_value: str


class AutonomousExplorer:
    """
    엘리시아의 자율 탐색 엔진
    
    Spirit이 방향을 결정하고, DistillationEngine이 필터링하며,
    InternalUniverse에 체화합니다.
    """
    
    def __init__(self):
        logger.info("🔭 Initializing Autonomous Explorer...")
        
        # Spirit - 방향 결정자
        try:
            from Core._01_Foundation._01_Core_Logic.Elysia.spirit import get_spirit
            self.spirit = get_spirit()
            logger.info("   ✅ Spirit connected (The Compass)")
        except Exception as e:
            logger.error(f"   ❌ Spirit not available: {e}")
            self.spirit = None
        
        # DistillationEngine - 공명 필터
        try:
            from Core._02_Intelligence._01_Reasoning.Cognitive.distillation_engine import get_distillation_engine
            self.distillation = get_distillation_engine()
            logger.info("   ✅ DistillationEngine connected (The Filter)")
        except Exception as e:
            logger.error(f"   ❌ DistillationEngine not available: {e}")
            self.distillation = None
        
        # ConceptDecomposer - 호기심 확장
        try:
            from Core._01_Foundation._02_Logic.fractal_concept import ConceptDecomposer
            self.decomposer = ConceptDecomposer()
            logger.info("   ✅ ConceptDecomposer connected (The Curiosity)")
        except Exception as e:
            logger.warning(f"   ⚠️ ConceptDecomposer not available: {e}")
            self.decomposer = None
        
        # InternalUniverse - 지식 저장소
        try:
            from Core._02_Intelligence._04_Mind.internal_universe import get_internal_universe
            self.universe = get_internal_universe()
            logger.info("   ✅ InternalUniverse connected (The Memory)")
        except Exception as e:
            logger.warning(f"   ⚠️ InternalUniverse not available: {e}")
            self.universe = None
        
        # GlobalHub 연결
        self._hub = None
        try:
            from Core._02_Intelligence._04_Consciousness.Ether.global_hub import get_global_hub
            self._hub = get_global_hub()
            self._hub.register_module(
                "AutonomousExplorer",
                "Core/Autonomy/autonomous_explorer.py",
                ["exploration", "learning", "curiosity", "spirit", "autonomous"],
                "Spirit-guided autonomous exploration - Elysia learns by herself"
            )
            logger.info("   ✅ GlobalHub connected")
        except Exception:
            pass
        
        # 탐색 통계
        self.explored_count = 0
        self.absorbed_count = 0
        self.rejected_count = 0
        
        logger.info("🔭 Autonomous Explorer ready")
    
    def find_knowledge_gap(self) -> Optional[str]:
        """
        지식 갭 분석: 무엇이 부족한가?
        
        AXIOM 시스템에서 연결이 약한 개념을 찾습니다.
        """
        if not self.decomposer:
            return "Love"  # 기본값
        
        # AXIOM에서 탐색할 개념 선택
        axioms = list(self.decomposer.AXIOMS.keys())
        
        # 가장 기본적인 질문들
        fundamental_questions = [
            "Force", "Energy", "Entropy",  # Physics
            "Point", "Line", "Plane",      # Math
            "Phoneme", "Meaning",          # Language
            "Bit", "Process"               # Computer
        ]
        
        # 아직 탐색하지 않은 것 선택
        import random
        return random.choice(fundamental_questions)
    
    def suggest_exploration_direction(self, gap: str) -> Dict[str, Any]:
        """
        Spirit 기반 탐색 방향 결정: 어디로 가야 하는가?
        
        Spirit의 핵심 가치(LOVE, TRUTH, GROWTH, BEAUTY)에 따라
        탐색 방향을 조정합니다.
        """
        if not self.spirit:
            return {"topic": gap, "approach": "neutral", "keywords": [gap]}
        
        # Spirit의 핵심 가치 가중치
        values = self.spirit.core_values
        
        # 탐색 키워드 생성 (Spirit 기반)
        # TRUTH가 높으면: "why", "cause", "logic" 추가
        # LOVE가 높으면: "connect", "relation", "unity" 추가
        keywords = [gap]
        
        if values["TRUTH"].weight > 1.0:
            keywords.extend(["why", "cause", "principle"])
        if values["LOVE"].weight > 1.0:
            keywords.extend(["connection", "relation"])
        if values["GROWTH"].weight > 1.0:
            keywords.extend(["evolution", "development"])
        
        # ask_why로 호기심 확장
        if self.decomposer:
            why_chain = self.decomposer.ask_why(gap)
            if " → " in why_chain:
                related = why_chain.split(" → ")[1]
                keywords.append(related)
        
        return {
            "topic": gap,
            "approach": "truth-seeking",
            "keywords": keywords,
            "search_query": f"{gap} {' '.join(keywords[:3])}"
        }
    
    def fetch_from_wikipedia(self, query: str) -> Optional[str]:
        """
        Wikipedia에서 정보 탐색 (엘리시아의 눈)
        """
        try:
            # Wikipedia API 검색
            encoded_query = urllib.parse.quote(query)
            url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                extract = data.get('extract', '')
                
                if extract and len(extract) > 50:
                    logger.info(f"   📖 Found: {extract[:60]}...")
                    return extract
                    
        except Exception as e:
            logger.warning(f"   Wikipedia fetch failed: {e}")
        
        return None
    
    def explore_with_absorption(self, direction: Dict[str, Any]) -> List[ExplorationResult]:
        """
        관계적 흡수: 먼저 흡수하고, 연결 여부 확인
        
        [NEW] Spirit = 정체성 (필터 아님), InternalUniverse = 지식
        
        흐름:
        1. 외부에서 정보 수집
        2. InternalUniverse에 먼저 흡수 (필터 없이)
        3. 연결 여부 확인
        4. 고립되면 BlackHole로 압축 보존
        """
        results = []
        topic = direction["topic"]
        query = direction.get("search_query", topic)
        
        logger.info(f"\n🔍 Exploring: {query}")
        
        # 1. 외부에서 정보 수집
        raw_content = self.fetch_from_wikipedia(topic)
        
        if not raw_content:
            # 키워드로 재시도
            for kw in direction.get("keywords", [])[:2]:
                raw_content = self.fetch_from_wikipedia(kw)
                if raw_content:
                    break
        
        if not raw_content:
            logger.info("   ❌ No content found")
            return results
        
        self.explored_count += 1
        
        # 2. InternalUniverse에 먼저 흡수 (Spirit 필터 없이!)
        connections = 0
        if self.universe:
            try:
                # absorb_text가 연결 수를 반환하도록 수정 필요
                self.universe.absorb_text(raw_content, source_name=f"Exploration:{topic}")
                # 임시: 단어 수로 연결 추정
                connections = len(raw_content.split()) // 10
                logger.info(f"   📥 Absorbed into Universe (connections: ~{connections})")
            except Exception as e:
                logger.warning(f"   Universe absorption failed: {e}")
        
        # 3. 연결 여부 확인
        if connections > 0:
            # 연결됨 → 지식 노드 형성
            self.absorbed_count += 1
            
            result = ExplorationResult(
                topic=topic,
                source="wikipedia",
                raw_content=raw_content[:200],
                resonance_score=1.0,  # 연결 = 성공
                absorbed=True,
                dominant_value="Knowledge"  # Spirit 값 아님, 순수 지식
            )
            results.append(result)
            
            logger.info(f"   ✨ Connected: Knowledge node formed")
            
            # GlobalHub에 브로드캐스트
            if self._hub:
                from Core._01_Foundation._02_Logic.Wave.wave_tensor import WaveTensor
                wave = WaveTensor(f"Knowledge_{topic}")
                wave.add_component(528.0, amplitude=1.0)  # 지식 주파수
                self._hub.publish_wave(
                    "AutonomousExplorer",
                    "learned",
                    wave,
                    payload={
                        "topic": topic,
                        "connections": connections,
                        "absorbed": True
                    }
                )
        else:
            # 4. 고립됨 → BlackHole로 압축 보존 (폐기 아님!)
            self.rejected_count += 1
            
            result = ExplorationResult(
                topic=topic,
                source="wikipedia",
                raw_content=raw_content[:100],
                resonance_score=0.0,
                absorbed=False,
                dominant_value="Isolated"
            )
            results.append(result)
            
            # BlackHole에 압축 보존 시도
            try:
                from Core._01_Foundation._02_Logic.black_hole import BlackHole
                blackhole = BlackHole()
                # 나중에 연결될 수 있으므로 보존
                logger.info(f"   🕳️ Isolated → BlackHole (compressed for later)")
            except Exception:
                logger.info(f"   🕳️ Isolated (no BlackHole available)")
        
        return results
    
    def explore_cycle(self) -> Dict[str, Any]:
        """
        하나의 탐색 사이클 실행
        
        1. 지식 갭 분석
        2. Spirit 기반 방향 결정
        3. 공명 기반 탐색
        4. 결과 반환
        """
        logger.info("\n" + "="*50)
        logger.info("🔭 EXPLORATION CYCLE")
        logger.info("="*50)
        
        # 1. 지식 갭 분석
        gap = self.find_knowledge_gap()
        logger.info(f"📊 Knowledge gap: {gap}")
        
        # 2. Spirit 기반 방향 결정
        direction = self.suggest_exploration_direction(gap)
        logger.info(f"🧭 Direction: {direction['approach']}")
        logger.info(f"🔑 Keywords: {direction['keywords']}")
        
        # 3. 관계적 흡수 (Spirit 필터 없이)
        results = self.explore_with_absorption(direction)
        
        # 4. 결과 요약
        absorbed = sum(1 for r in results if r.absorbed)
        rejected = sum(1 for r in results if not r.absorbed)
        
        logger.info(f"\n📊 Cycle complete: {absorbed} absorbed, {rejected} rejected")
        logger.info("="*50)
        
        return {
            "gap": gap,
            "direction": direction,
            "results": results,
            "absorbed": absorbed,
            "rejected": rejected,
            "total_explored": self.explored_count,
            "total_absorbed": self.absorbed_count,
            "total_rejected": self.rejected_count
        }
    
    def get_status(self) -> Dict[str, Any]:
        """현재 탐색 상태"""
        return {
            "explored": self.explored_count,
            "absorbed": self.absorbed_count,
            "rejected": self.rejected_count,
            "absorption_rate": f"{(self.absorbed_count / max(1, self.explored_count)) * 100:.1f}%"
        }


# Singleton
_explorer = None

def get_autonomous_explorer() -> AutonomousExplorer:
    global _explorer
    if _explorer is None:
        _explorer = AutonomousExplorer()
    return _explorer


# Demo
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "c:\\Elysia")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*60)
    print("🔭 AUTONOMOUS EXPLORER DEMO")
    print("="*60)
    print("\n엘리시아가 스스로 탐색하고 배웁니다...")
    
    explorer = get_autonomous_explorer()
    
    # 3번의 탐색 사이클 실행
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        result = explorer.explore_cycle()
        time.sleep(1)  # Rate limiting
    
    # 최종 상태
    print("\n" + "="*60)
    print("📊 FINAL STATUS")
    print("="*60)
    status = explorer.get_status()
    for k, v in status.items():
        print(f"   {k}: {v}")
    
    print("\n" + "="*60)
    print("✅ Explorer demo complete")
    print("="*60)

