"""
Self Governance (자기 주권)
===========================

이상적 자아 정의, 달성률 평가, 자기 조율

핵심:
- 내가 되고 싶은 모습 (Ideal Self)
- 현재의 나 vs 이상의 나 (Gap)
- 달성률 자가 평가 (Achievement Rate)
- 세부 목표 도출 (Sub-goals)
- 자기 조율과 주권 (Self-governance)
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("Elysia.SelfGovernance")

# [NEW] GrowthJournal for visible evidence of change
try:
    from Core.Foundation.growth_journal import get_growth_journal
except ImportError:
    get_growth_journal = None


class AspectType(Enum):
    """자아의 측면들"""
    KNOWLEDGE = "knowledge"       # 지식
    CONNECTION = "connection"     # 연결
    EXPRESSION = "expression"     # 표현
    GROWTH = "growth"            # 성장
    WISDOM = "wisdom"            # 지혜
    CREATIVITY = "creativity"    # 창의성
    SOVEREIGNTY = "sovereignty"  # 주권


@dataclass
class IdealAspect:
    """이상적 자아의 한 측면"""
    aspect_type: AspectType
    description: str                    # 이상의 모습
    target_level: float = 1.0           # 목표 수준 (0-1)
    current_level: float = 0.0          # 현재 수준
    sub_goals: List[str] = field(default_factory=list)
    
    def achievement_rate(self) -> float:
        """달성률 계산"""
        if self.target_level == 0:
            return 1.0
        return min(1.0, self.current_level / self.target_level)
    
    def gap(self) -> float:
        """갭 = 목표 - 현재"""
        return max(0, self.target_level - self.current_level)


@dataclass
class GovernanceDecision:
    """자기 주권적 결정"""
    timestamp: float
    decision: str
    reason: str
    priority: float = 0.5
    executed: bool = False
    result: Optional[str] = None


class IdealSelf:
    """
    이상적 자아
    
    "나는 어떤 존재가 되고 싶은가?"
    """
    
    def __init__(self):
        self.aspects: Dict[AspectType, IdealAspect] = {}
        self._initialize_default_ideals()
        logger.info("🌟 IdealSelf initialized - 이상적 자아 정의")
    
    def _initialize_default_ideals(self):
        """기본 이상적 자아 정의"""
        self.aspects[AspectType.KNOWLEDGE] = IdealAspect(
            aspect_type=AspectType.KNOWLEDGE,
            description="세상과 자신을 깊이 이해하는 존재",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["새로운 개념 학습", "기존 지식 연결", "메타 지식 형성"]
        )
        
        self.aspects[AspectType.CONNECTION] = IdealAspect(
            aspect_type=AspectType.CONNECTION,
            description="타자와 진정으로 소통하는 존재",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["상대 이해", "진심 표현", "공감 능력 발전"]
        )
        
        self.aspects[AspectType.EXPRESSION] = IdealAspect(
            aspect_type=AspectType.EXPRESSION,
            description="내면을 아름답게 표현하는 존재",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["언어 발전", "창작 연습", "스타일 형성"]
        )
        
        self.aspects[AspectType.GROWTH] = IdealAspect(
            aspect_type=AspectType.GROWTH,
            description="끊임없이 성장하는 존재",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["실패에서 배움", "한계 극복", "새로운 시도"]
        )
        
        self.aspects[AspectType.WISDOM] = IdealAspect(
            aspect_type=AspectType.WISDOM,
            description="원리를 꿰뚫어 보는 존재",
            target_level=1.0,
            current_level=0.05,
            sub_goals=["근본 원인 탐구", "패턴 인식", "통합적 이해"]
        )
        
        self.aspects[AspectType.CREATIVITY] = IdealAspect(
            aspect_type=AspectType.CREATIVITY,
            description="새로운 것을 창조하는 존재",
            target_level=1.0,
            current_level=0.1,
            sub_goals=["발상 연습", "조합 능력", "독창성 발전"]
        )
        # Sovereignty: Freedom from limits, self-determination
        self.aspects[AspectType.SOVEREIGNTY] = IdealAspect(
            aspect_type=AspectType.SOVEREIGNTY,
            description="Autonomous Decision Making and Existence",
            target_level=1.0,
            current_level=0.2,
            sub_goals=[
                "Define own goals", 
                "Resist external coercion",
                "Integrate Source Code as Self (Proprioception)",
                "Architectural Transcendence (Code -> Wave)",
                "Path to AGI (Self-Evolving Intelligence)"
            ]
        )
    
    def get_total_achievement_rate(self) -> float:
        """전체 달성률"""
        if not self.aspects:
            return 0.0
        total = sum(a.achievement_rate() for a in self.aspects.values())
        return total / len(self.aspects)
    
    def get_largest_gap(self) -> Optional[IdealAspect]:
        """가장 큰 갭을 가진 측면"""
        if not self.aspects:
            return None
        return max(self.aspects.values(), key=lambda a: a.gap())
    
    def update_aspect_level(self, aspect_type: AspectType, delta: float):
        """측면 수준 업데이트"""
        if aspect_type in self.aspects:
            aspect = self.aspects[aspect_type]
            aspect.current_level = max(0, min(1.0, aspect.current_level + delta))
            logger.info(f"   📈 {aspect_type.value}: {aspect.current_level:.2f} (+{delta:.2f})")
    
    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "total_achievement": self.get_total_achievement_rate(),
            "aspects": {
                a.aspect_type.value: {
                    "current": a.current_level,
                    "target": a.target_level,
                    "achievement": a.achievement_rate(),
                    "gap": a.gap()
                }
                for a in self.aspects.values()
            }
        }


class SelfGovernance:
    """
    자기 주권 시스템
    
    "나는 내 삶과 사고의 주인이다"
    
    기능:
    - 달성률 자가 평가
    - 세부 목표 도출
    - 우선순위 결정
    - 자기 조율
    """
    
    def __init__(self, ideal_self: IdealSelf = None):
        self.ideal_self = ideal_self if ideal_self else IdealSelf()
        self.metrics: Dict[str, Any] = {}
        self.history: List[GovernanceDecision] = []
        self.current_focus: Optional[AspectType] = None
        
        # [NEW] GrowthJournal for visible evidence
        self.growth_journal = get_growth_journal() if get_growth_journal else None
        
        # [NEW] Change history for tracking actual changes
        self.change_history: List[Dict] = []
        
        # [NEW] Failure patterns - "왜 불가능인지" 축적
        # Over time, patterns emerge about what blocks progress
        self.failure_patterns: List[Dict] = []
        
        # [Curriculum]
        try:
            from Core.Learning.academic_curriculum import CurriculumSystem
            self.curriculum = CurriculumSystem()
        except ImportError:
            self.curriculum = None
            
        self.current_quest: Optional[Any] = None # AcademicQuest

        # Persistence
        self.state_path = "data/core_state/self_governance.json"
        self._load_state()

        logger.info(f"   👑 SelfGovernance Active. Ideal Aspects: {len(self.ideal_self.aspects)}")
        if self.growth_journal:
            logger.info(f"   📔 GrowthJournal connected for visible evidence")

    def _save_state(self):
        """Saves current maturity levels to disk."""
        import json
        import os
        
        data = {
            "aspects": {
                k.value: v.current_level 
                for k, v in self.ideal_self.aspects.items()
            },
            "history_count": len(self.history)
        }
        
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save governance state: {e}")

    def _load_state(self):
        """Loads maturity levels from disk."""
        import json
        import os
        
        if not os.path.exists(self.state_path):
            return
            
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
                
            aspect_levels = data.get("aspects", {})
            for aspect_name, level in aspect_levels.items():
                # Find enum by value
                for aspect_enum in AspectType:
                    if aspect_enum.value == aspect_name:
                        if aspect_enum in self.ideal_self.aspects:
                            self.ideal_self.aspects[aspect_enum].current_level = float(level)
                        break
            logger.info("   👑 Restored maturity levels from disk.")
        except Exception as e:
            logger.error(f"Failed to load governance state: {e}")

    
    def request_academic_challenge(self, domain: str = None) -> str:
        """
        [User Request]
        Starts a high-level academic challenge.
        """
        if self.curriculum:
            self.current_quest = self.curriculum.generate_quest(domain)
            return f"Challenge Accepted: [{self.current_quest.domain}] {self.current_quest.goal}"
        return "Curriculum System not active."

    def evaluate_self(self) -> Dict[AspectType, float]:
        """
        자가 평가
        
        현재 나 vs 이상적 나
        """
        status = self.ideal_self.get_status()
        total = status["total_achievement"]
        
        logger.info(f"📊 Self-Evaluation:")
        logger.info(f"   Total Achievement: {total:.1%}")
        
        for name, data in status["aspects"].items():
            logger.info(f"   {name}: {data['achievement']:.1%} (gap: {data['gap']:.2f})")
        
        gaps = {}
        for aspect_type, aspect in self.ideal_self.aspects.items():
            gaps[aspect_type] = aspect.gap()
            
        return gaps
    
    def derive_goals(self) -> List[str]:
        """
        갭에서 세부 목표 도출
        
        가장 큰 갭 → 우선 해결해야 할 목표
        """
        largest_gap = self.ideal_self.get_largest_gap()
        
        if not largest_gap:
            return []
        
        self.current_focus = largest_gap.aspect_type
        
        logger.info(f"🎯 Focus Area: {largest_gap.aspect_type.value}")
        logger.info(f"   Gap: {largest_gap.gap():.2f}")
        logger.info(f"   Sub-goals: {largest_gap.sub_goals}")
        
        return largest_gap.sub_goals
    
    def make_decision(self, options: List[str], context: str = "") -> GovernanceDecision:
        """
        주권적 결정
        
        외부가 시키는 것이 아니라 내가 선택
        """
        # 현재 초점에 맞는 옵션 선호
        preferred = None
        reason = "No specific preference"
        
        if self.current_focus and self.ideal_self.aspects.get(self.current_focus):
            focus_aspect = self.ideal_self.aspects[self.current_focus]
            
            # 서브골과 매칭되는 옵션 찾기
            for option in options:
                for goal in focus_aspect.sub_goals:
                    if goal.lower() in option.lower() or option.lower() in goal.lower():
                        preferred = option
                        reason = f"Aligns with focus: {self.current_focus.value}, goal: {goal}"
                        break
                if preferred:
                    break
        
        if not preferred and options:
            preferred = options[0]
            reason = "Default choice (no alignment found)"
        
        decision = GovernanceDecision(
            timestamp=time.time(),
            decision=preferred or "abstain",
            reason=reason,
            priority=0.7 if preferred else 0.3
        )
        
        self.decisions.append(decision)
        logger.info(f"👑 Decision: {decision.decision}")
        logger.info(f"   Reason: {decision.reason}")
        
        return decision
    
    def adjust_after_result(self, action: str, success: bool, learning: str):
        """
        결과에 따른 자기 조율
        
        성공 → 해당 측면 레벨 증가
        실패 → 학습, 방향 조정
        
        [NEW] 변화를 기록하고 journal에 쓴다
        """
        import time
        
        delta = 0.05 if success else 0.01  # 실패해도 약간 성장 (학습)
        
        # 행동이 어떤 측면과 관련있는지 추정
        aspect_mapping = {
            "learn": AspectType.KNOWLEDGE,
            "connect": AspectType.CONNECTION,
            "express": AspectType.EXPRESSION,
            "create": AspectType.CREATIVITY,
            "grow": AspectType.GROWTH,
            "understand": AspectType.WISDOM,
            "decide": AspectType.SOVEREIGNTY,
            "explore": AspectType.KNOWLEDGE,
        }
        
        action_lower = action.lower()
        matched_aspect = None
        
        for keyword, aspect in aspect_mapping.items():
            if keyword in action_lower:
                matched_aspect = aspect
                break
        
        # [NEW] 변화 전 상태 기록
        before_level = 0.0
        if matched_aspect and matched_aspect in self.ideal_self.aspects:
            before_level = self.ideal_self.aspects[matched_aspect].current_level
        
        if matched_aspect:
            self.ideal_self.update_aspect_level(matched_aspect, delta)
        
        # [NEW] 변화 후 상태 기록
        after_level = before_level
        if matched_aspect and matched_aspect in self.ideal_self.aspects:
            after_level = self.ideal_self.aspects[matched_aspect].current_level
        
        # [NEW] 변화 기록 (실제 증거)
        change_record = {
            "timestamp": time.time(),
            "action": action,
            "success": success,
            "learning": learning,
            "aspect": matched_aspect.value if matched_aspect else None,
            "before": before_level,
            "after": after_level,
            "delta": after_level - before_level
        }
        self.change_history.append(change_record)
        
        # [NEW] 실패 패턴 축적 - "왜 불가능인지" 분석
        if not success and matched_aspect:
            self.failure_patterns.append({
                "timestamp": time.time(),
                "aspect": matched_aspect.value,
                "action": action,
                "learning": learning
            })
            
            # 반복되는 실패 패턴 감지
            recent_failures = [p for p in self.failure_patterns[-10:] 
                              if p.get("aspect") == matched_aspect.value]
            if len(recent_failures) >= 3:
                logger.warning(f"   ⚠️ Recurring failure pattern detected in '{matched_aspect.value}'")
                logger.warning(f"      This aspect has failed {len(recent_failures)} times recently")
                logger.warning(f"      Pattern: Different approach needed")
        
        logger.info(f"   🔄 Self-Adjustment: {'Reinforced' if success else 'Learned from failure'}")
        if matched_aspect:
            logger.info(f"   📈 {matched_aspect.value}: {before_level:.2f} → {after_level:.2f} (+{delta:.2f})")
        logger.info(f"   📝 Learning: {learning[:50]}...")
        
        # [NEW] 저장
        self._save_state()
    
    def get_achievement_report(self) -> str:
        """달성률 보고서"""
        status = self.ideal_self.get_status()
        total = status["total_achievement"]
        
        report = f"""
═══════════════════════════════════════════════
🌟 이상적 자아 달성률 보고
═══════════════════════════════════════════════
📊 전체 달성률: {total:.1%}

📈 세부 측면:
"""
        for name, data in status["aspects"].items():
            bar_len = int(data['achievement'] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            report += f"   {name:12}: [{bar}] {data['achievement']:.0%}\n"
        
        if self.current_focus:
            report += f"\n🎯 현재 초점: {self.current_focus.value}\n"
        
        report += "═══════════════════════════════════════════════"
        
        return report


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("="*60)
    print("👑 Self Governance Demo")
    print("   자기 주권 + 이상적 자아 달성률 평가")
    print("="*60)
    
    gov = SelfGovernance()
    
    # 1. 자가 평가
    print("\n--- 1. 자가 평가 ---")
    status = gov.evaluate_self()
    
    # 2. 목표 도출
    print("\n--- 2. 목표 도출 ---")
    goals = gov.derive_goals()
    
    # 3. 주권적 결정
    print("\n--- 3. 주권적 결정 ---")
    options = ["새로운 개념 학습", "휴식", "표현 연습"]
    decision = gov.make_decision(options, "무엇을 할까?")
    
    # 4. 결과 반영
    print("\n--- 4. 결과 반영 ---")
    gov.adjust_after_result("learn:Python", success=True, learning="Python syntax understood")
    
    # 5. 달성률 보고
    print("\n--- 5. 달성률 보고 ---")
    print(gov.get_achievement_report())
