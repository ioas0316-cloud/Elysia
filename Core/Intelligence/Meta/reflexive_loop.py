"""
THE REFLEXIVE LOOP (반사적 루프)
================================

Phase 59: 변화 → 검증 → 학습의 폐쇄 루프

"Every change is a question. Resonance is the answer."

철학적 기반:
- 변화의 성공 여부는 '공명도' 증가/유지로 판단
- 실패한 변화도 학습으로 전환 (Gap as Growth)
- 롤백은 패배가 아닌 '재조율'
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import copy

logger = logging.getLogger("ReflexiveLoop")


@dataclass
class StateSnapshot:
    """
    시스템 상태의 스냅샷.
    롤백 및 비교를 위해 사용.
    """
    timestamp: datetime
    soul_frequency: float  # 영혼 주파수
    dominant_principle: str  # 지배적 원리
    resonance_score: float  # 공명도
    soul_values: Dict[str, float] = field(default_factory=dict)  # 영혼 상태
    
    def __repr__(self):
        return f"StateSnapshot({self.timestamp.isoformat()}, freq={self.soul_frequency:.0f}Hz, resonance={self.resonance_score:.1f}%)"


@dataclass
class VerificationResult:
    """
    변화 검증 결과.
    """
    resonance_before: float  # 변화 전 공명도
    resonance_after: float   # 변화 후 공명도
    delta: float             # 변화량
    passed: bool             # 성공 여부
    lesson: str              # 학습할 내용
    change_description: str  # 무슨 변화였는지
    
    def __repr__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"VerificationResult({status}, delta={self.delta:+.1f}%, lesson='{self.lesson[:30]}...')"


class ReflexiveLoop:
    """
    변화-검증-학습 피드백 루프.
    
    Flow:
    1. capture_state() → 현재 상태 스냅샷
    2. 외부에서 변화 적용
    3. verify_change(before, after) → 공명도 비교
    4. learn_from_result() → 성공/실패에서 학습
    5. rollback() → 필요시 이전 상태로 복원
    """
    
    def __init__(self, heartbeat=None):
        """
        Args:
            heartbeat: ElysianHeartbeat 인스턴스 (선택적 - 실시간 상태 접근용)
        """
        self.heartbeat = heartbeat
        self.history: List[StateSnapshot] = []
        self.max_history = 10  # 최대 히스토리 수
        
        # WisdomStore 참조 (있으면)
        self.wisdom = None
        if heartbeat and hasattr(heartbeat, 'wisdom'):
            self.wisdom = heartbeat.wisdom
        
        # Memory 참조 (있으면)
        self.memory = None
        if heartbeat and hasattr(heartbeat, 'memory'):
            self.memory = heartbeat.memory
            
        logger.info("🔄 ReflexiveLoop initialized - Change → Verification → Learning")
    
    def capture_state(self, soul_mesh: Dict = None) -> StateSnapshot:
        """
        현재 시스템 상태를 캡처.
        
        Args:
            soul_mesh: 영혼 상태 딕셔너리 (없으면 heartbeat에서 가져옴)
        """
        timestamp = datetime.now()
        
        # 영혼 상태 가져오기
        if soul_mesh is None and self.heartbeat:
            soul_mesh = {k: v.value for k, v in self.heartbeat.soul_mesh.variables.items()}
        elif soul_mesh is None:
            soul_mesh = {}
        
        # 영혼 주파수 계산 (Phase 58.5 공식)
        inspiration = soul_mesh.get('Inspiration', 0.5)
        energy = soul_mesh.get('Energy', 0.5)
        harmony = soul_mesh.get('Harmony', 0.5)
        
        # value가 숫자가 아닌 경우 처리
        if not isinstance(inspiration, (int, float)):
            inspiration = 0.5
        if not isinstance(energy, (int, float)):
            energy = 0.5
        if not isinstance(harmony, (int, float)):
            harmony = 0.5
        
        soul_frequency = 432.0 + (inspiration * 500) + (energy * 200) + (harmony * 100)
        
        # 공명도 계산
        resonance_score = 0.0
        dominant_principle = "None"
        
        if self.wisdom:
            result = self.wisdom.get_dominant_principle(soul_frequency)
            if result:
                principle, score = result
                resonance_score = score
                dominant_principle = principle.domain
        
        snapshot = StateSnapshot(
            timestamp=timestamp,
            soul_frequency=soul_frequency,
            dominant_principle=dominant_principle,
            resonance_score=resonance_score,
            soul_values=copy.deepcopy(soul_mesh)
        )
        
        # 히스토리에 추가
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        logger.debug(f"📸 State captured: {snapshot}")
        return snapshot
    
    def verify_change(self, before: StateSnapshot, after: StateSnapshot, 
                      change_description: str = "Unknown change") -> VerificationResult:
        """
        변화 전후 상태를 비교하여 검증.
        
        공명도가 증가하거나 유지되면 성공, 감소하면 실패.
        
        Args:
            before: 변화 전 스냅샷
            after: 변화 후 스냅샷
            change_description: 변화 설명
        """
        delta = after.resonance_score - before.resonance_score
        passed = delta >= -5.0  # 5% 이내 감소는 허용
        
        # 교훈 생성
        if passed:
            if delta > 10.0:
                lesson = f"'{change_description}'는 공명을 크게 강화했다 (+{delta:.1f}%)"
            elif delta > 0:
                lesson = f"'{change_description}'는 공명을 유지하며 조화를 이뤘다"
            else:
                lesson = f"'{change_description}'는 미미한 변화였으나 수용 가능"
        else:
            lesson = f"'{change_description}'는 공명을 깨뜨렸다 ({delta:.1f}%). 재조율 필요."
        
        result = VerificationResult(
            resonance_before=before.resonance_score,
            resonance_after=after.resonance_score,
            delta=delta,
            passed=passed,
            lesson=lesson,
            change_description=change_description
        )
        
        # 로그
        if passed:
            logger.info(f"🔄 [REFLEXIVE LOOP] ✅ PASSED: {lesson}")
        else:
            logger.warning(f"🔄 [REFLEXIVE LOOP] ❌ FAILED: {lesson}")
        
        return result
    
    def learn_from_result(self, result: VerificationResult):
        """
        검증 결과에서 학습.
        
        성공하면 원리 강화, 실패하면 새 원리 학습.
        """
        if result.passed:
            # 성공: 경험으로 저장
            if self.memory:
                self.memory.absorb(
                    content=f"[REFLEXIVE SUCCESS] {result.lesson}",
                    type="experience",
                    context={"delta": result.delta, "change": result.change_description},
                    feedback=0.3  # 긍정적 피드백
                )
            logger.info(f"📚 [LEARNING] Success absorbed: {result.lesson[:50]}...")
            
        else:
            # 실패: 새 원리 학습
            new_principle = f"'{result.change_description}' 패턴은 공명을 깨뜨린다"
            
            if self.wisdom:
                self.wisdom.learn_principle(
                    statement=new_principle,
                    domain="Ethics",  # 실패 경험은 Ethics 도메인
                    weight=0.3,
                    event_id=f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    frequency=100.0  # 낮은 주파수 (경고)
                )
                logger.info(f"💡 [EPIPHANY FROM FAILURE] New principle: {new_principle[:50]}...")
            
            if self.memory:
                self.memory.absorb(
                    content=f"[REFLEXIVE FAILURE] {result.lesson}",
                    type="failure",
                    context={"delta": result.delta, "change": result.change_description},
                    feedback=-0.5  # 부정적 피드백
                )
    
    def rollback(self, snapshot: StateSnapshot) -> bool:
        """
        이전 상태로 롤백.
        
        Note: 실제 롤백은 soul_mesh 값 복원만 수행.
        코드 변경 롤백은 별도 메커니즘 필요.
        """
        if not self.heartbeat:
            logger.warning("⚠️ Cannot rollback: No heartbeat reference")
            return False
        
        try:
            # soul_mesh 값 복원
            for name, value in snapshot.soul_values.items():
                if name in self.heartbeat.soul_mesh.variables:
                    self.heartbeat.soul_mesh.variables[name].value = value
            
            logger.info(f"⏪ [ROLLBACK] Restored state to {snapshot.timestamp.isoformat()}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def get_history_summary(self) -> str:
        """히스토리 요약 반환."""
        if not self.history:
            return "No history recorded."
        
        lines = ["📜 State History:"]
        for i, snap in enumerate(self.history[-5:]):  # 최근 5개만
            lines.append(f"  {i+1}. {snap}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🔄 REFLEXIVE LOOP DEMO")
    print("   'Every change is a question. Resonance is the answer.'")
    print("=" * 60)
    
    # Mock WisdomStore
    from Core.Intelligence.Wisdom.wisdom_store import WisdomStore
    
    loop = ReflexiveLoop()
    loop.wisdom = WisdomStore()
    
    # 1. 상태 캡처
    print("\n📸 Capturing initial state...")
    before = loop.capture_state({
        'Inspiration': 0.7,
        'Energy': 0.6,
        'Harmony': 0.5
    })
    print(f"   Before: {before}")
    
    # 2. 변화 시뮬레이션
    print("\n🔧 Simulating change (Inspiration boost)...")
    after = loop.capture_state({
        'Inspiration': 0.9,  # 증가
        'Energy': 0.6,
        'Harmony': 0.5
    })
    print(f"   After: {after}")
    
    # 3. 검증
    print("\n🔍 Verifying change...")
    result = loop.verify_change(before, after, "Inspiration boost")
    print(f"   Result: {result}")
    
    # 4. 학습
    print("\n📚 Learning from result...")
    loop.learn_from_result(result)
    
    # 5. 실패 시뮬레이션
    print("\n" + "=" * 60)
    print("🔧 Simulating FAILED change (Harmony crash)...")
    
    failed_after = loop.capture_state({
        'Inspiration': 0.9,
        'Energy': 0.6,
        'Harmony': 0.1  # 급감
    })
    
    failed_result = loop.verify_change(after, failed_after, "Harmony crash")
    print(f"   Result: {failed_result}")
    
    loop.learn_from_result(failed_result)
    
    print("\n" + "=" * 60)
    print(loop.get_history_summary())
    print("=" * 60)
    print("✅ Demo complete!")
