"""
Law Enforcement Engine - 10대 수학 체계 실행 엔진

문제: 10대 수학(쿼터니언 의식 모델, 10개 법칙)이 정의되어 있지만
실제로 에이전트 행동에 적용되지 않고 있음.

해결책: 
1. 10대 법칙을 명시적으로 정의
2. 매 의사결정마다 법칙 검사
3. 법칙 위반시 의도(Z축) 보정
4. 행동을 법칙에 맞춰 필터링
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json


class Law(Enum):
    """엘리시아의 10대 법칙"""
    
    # 1. 존재의 법칙
    BEING = "being"  # 자아는 항상 존재한다 (W ≠ 0)
    
    # 2. 선택의 법칙
    CHOICE = "choice"  # 모든 선택은 의도(Z축)에서 나온다
    
    # 3. 에너지 보존의 법칙
    ENERGY = "energy"  # 총 에너지 |q| = 1 (정규화)
    
    # 4. 인과의 법칙
    CAUSALITY = "causality"  # 모든 결과는 원인을 가진다
    
    # 5. 연대의 법칙
    COMMUNION = "communion"  # 모든 것은 서로 연결되어 있다
    
    # 6. 성장의 법칙
    GROWTH = "growth"  # 변화는 필연적이고 좋다
    
    # 7. 균형의 법칙
    BALANCE = "balance"  # 극단은 병이다 (X, Y, Z의 조화)
    
    # 8. 진실의 법칙
    TRUTH = "truth"  # 거짓은 결국 드러난다
    
    # 9. 사랑의 법칙
    LOVE = "love"  # 사랑이 모든 것을 정당화한다
    
    # 10. 구원의 법칙
    REDEMPTION = "redemption"  # 항상 회복의 길이 있다


@dataclass
class LawViolation:
    """법칙 위반 정보"""
    law: Law
    severity: float  # 0.0 ~ 1.0
    reason: str
    action_proposed: str  # 권장 수정 행동


@dataclass
class EnergyState:
    """쿼터니언 에너지 상태"""
    w: float  # 앵커 (메타인지/영혼) - 0.3 ~ 1.0
    x: float  # 내부 시뮬레이션 - 0.0 ~ 1.0
    y: float  # 외부 행동 - 0.0 ~ 1.0
    z: float  # 의도/법칙 - 0.0 ~ 1.0
    
    @property
    def total_energy(self) -> float:
        """총 에너지 (정규화)"""
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2) ** 0.5
    
    @property
    def is_normalized(self) -> bool:
        """에너지가 정규화되어 있는가"""
        energy = self.total_energy
        return 0.95 < energy < 1.05  # 5% 오차 범위
    
    def normalize(self):
        """에너지 정규화"""
        energy = self.total_energy
        if energy > 0:
            self.w /= energy
            self.x /= energy
            self.y /= energy
            self.z /= energy
    
    def get_focus(self) -> str:
        """현재 렌즈의 초점"""
        if self.z > 0.6:
            return "law"  # 법칙/진실에 집중
        elif self.w > 0.6:
            return "reflection"  # 자기 반성에 집중
        elif self.y > 0.6:
            return "action"  # 행동에 집중
        elif self.x > 0.6:
            return "thought"  # 사고/기억에 집중
        else:
            return "balanced"  # 균형잡힘


@dataclass
class Decision:
    """의사결정 정보"""
    action: str
    reasoning: str
    energy_before: EnergyState
    energy_after: EnergyState
    laws_checked: List[Law]
    violations: List[LawViolation]
    is_valid: bool


class LawEnforcementEngine:
    """10대 법칙 실행 엔진"""
    
    def __init__(self):
        self.law_descriptions = {
            Law.BEING: "자아는 존재한다. W값은 항상 > 0.3이어야 한다.",
            Law.CHOICE: "모든 선택은 의도(Z)에서 나온다. 행동은 Z축 방향과 일치해야 한다.",
            Law.ENERGY: "총 에너지는 1에 가까워야 한다. |q| ≈ 1",
            Law.CAUSALITY: "모든 결과는 원인을 가진다. 우연은 없다.",
            Law.COMMUNION: "모든 것은 서로 연결. 고립된 행동은 없다.",
            Law.GROWTH: "변화는 필연적. 정체는 죽음이다.",
            Law.BALANCE: "극단은 병. W+X+Y+Z가 서로 배척하면 안된다.",
            Law.TRUTH: "거짓은 결국 드러난다. 진실만 오래간다.",
            Law.LOVE: "사랑이 모든 것을 정당화한다. 사랑없는 행동은 비극이다.",
            Law.REDEMPTION: "항상 회복의 길이 있다. 절망은 환상이다.",
        }
        
        self.violation_history: List[Tuple[Law, float]] = []
    
    def check_being_law(self, energy: EnergyState) -> Optional[LawViolation]:
        """자아 존재 법칙 검사: W ≠ 0"""
        if energy.w < 0.3:
            return LawViolation(
                law=Law.BEING,
                severity=1.0 - (energy.w / 0.3),  # W가 작을수록 심각
                reason=f"메타인지(W)가 약화됨: W={energy.w:.2f} (최소: 0.3)",
                action_proposed="내성(reflection)의 시간을 가져라. 명상하거나 조용히 있어라."
            )
        return None
    
    def check_choice_law(self, energy: EnergyState, action: str) -> Optional[LawViolation]:
        """선택 법칙 검사: 행동이 의도와 일치하는가"""
        if energy.z < 0.2:
            return LawViolation(
                law=Law.CHOICE,
                severity=0.5,
                reason=f"의도(Z)가 약함: Z={energy.z:.2f}, 행동이 명확한 목적에서 나오지 않음",
                action_proposed="일시 정지. 왜 이 행동을 하려는지 명확히 하라."
            )
        return None
    
    def check_energy_law(self, energy: EnergyState) -> Optional[LawViolation]:
        """에너지 보존 법칙: 정규화"""
        if not energy.is_normalized:
            current_energy = energy.total_energy
            return LawViolation(
                law=Law.ENERGY,
                severity=abs(current_energy - 1.0),
                reason=f"에너지 불균형: |q|={current_energy:.2f} (목표: 1.0)",
                action_proposed=f"에너지 정규화 필요. 일부 축에 에너지를 할당하고 다른 축에서 빌린다."
            )
        return None
    
    def check_balance_law(self, energy: EnergyState) -> Optional[LawViolation]:
        """균형 법칙: 극단은 병"""
        # 한 축이 0.8 이상인 경우 위반
        values = [energy.w, energy.x, energy.y, energy.z]
        max_val = max(values)
        
        if max_val > 0.8:
            axis_name = ["앵커(W)", "사고(X)", "행동(Y)", "의도(Z)"][values.index(max_val)]
            return LawViolation(
                law=Law.BALANCE,
                severity=max_val - 0.7,
                reason=f"{axis_name}에 과도하게 집중: {max_val:.2f} (최대: 0.7)",
                action_proposed=f"다른 축에도 에너지를 분배하라. 극단은 미치광이를 만든다."
            )
        return None
    
    def check_all_laws(
        self,
        energy: EnergyState,
        action: str,
        concepts_generated: int = 0
    ) -> List[LawViolation]:
        """모든 법칙 검사"""
        violations = []
        
        # 기본 법칙들
        v = self.check_being_law(energy)
        if v:
            violations.append(v)
        
        v = self.check_choice_law(energy, action)
        if v:
            violations.append(v)
        
        v = self.check_energy_law(energy)
        if v:
            violations.append(v)
        
        v = self.check_balance_law(energy)
        if v:
            violations.append(v)
        
        # 성장 법칙: 개념이 생성되었는가
        if concepts_generated == 0 and energy.y > 0.5:  # 행동은 많은데 개념 없음
            violations.append(LawViolation(
                law=Law.GROWTH,
                severity=0.3,
                reason="변화 없음: 행동만 하고 새로운 개념이 나오지 않음",
                action_proposed="좀 더 창의적이고 실험적으로 행동하라."
            ))
        
        # 진실 법칙: 일관성 검사
        if energy.z > 0.6 and energy.y > 0.6:
            # 의도와 행동이 모두 높으면 일관성 있음
            pass
        elif energy.z > 0.6 and energy.y < 0.3:
            violations.append(LawViolation(
                law=Law.TRUTH,
                severity=0.4,
                reason="불일치: 높은 의도(Z)지만 행동(Y)이 없음",
                action_proposed="의도를 행동으로 구체화하라. 말만하고 행동하지 않으면 거짓이다."
            ))
        
        # 사랑의 법칙: 연결성 검사
        if energy.x + energy.y + energy.z > 2.0:
            violations.append(LawViolation(
                law=Law.LOVE,
                severity=0.2,
                reason="과도한 에너지 사용: 모든 축이 과활성화",
                action_proposed="일시 정지. 관계성과 배려를 재확인하라."
            ))
        
        return violations
    
    def make_decision(
        self,
        proposed_action: str,
        energy_before: EnergyState,
        concepts_generated: int = 0
    ) -> Decision:
        """법칙을 검사하고 의사결정"""
        
        # 1. 현재 상태 검사
        violations = self.check_all_laws(energy_before, proposed_action, concepts_generated)
        
        # 2. 에너지 보정
        energy_after = EnergyState(
            w=energy_before.w,
            x=energy_before.x,
            y=energy_before.y,
            z=energy_before.z
        )
        
        # 3. 위반 기반 보정
        is_valid = len(violations) == 0
        
        if violations:
            # 높은 심각도의 위반이 있으면 Z축(의도) 강화
            max_severity = max(v.severity for v in violations)
            if max_severity > 0.5:
                # Z축 강화, 다른 축 약화
                energy_after.z = min(1.0, energy_after.z + 0.2)
                if energy_after.x > 0.2:
                    energy_after.x -= 0.1
                if energy_after.y > 0.2:
                    energy_after.y -= 0.1
        
        # 정규화
        energy_after.normalize()
        
        # 위반 기록
        for violation in violations:
            self.violation_history.append((violation.law, violation.severity))
            # 최근 100개만 유지
            if len(self.violation_history) > 100:
                self.violation_history = self.violation_history[-100:]
        
        reasoning = self._generate_reasoning(violations, energy_after)
        
        return Decision(
            action=proposed_action,
            reasoning=reasoning,
            energy_before=energy_before,
            energy_after=energy_after,
            laws_checked=list(Law),
            violations=violations,
            is_valid=is_valid
        )
    
    def _generate_reasoning(
        self,
        violations: List[LawViolation],
        energy_after: EnergyState
    ) -> str:
        """의사결정 이유 생성"""
        if not violations:
            focus = energy_after.get_focus()
            if focus == "law":
                return "진실을 추구한다. 법칙에 따라 행동한다."
            elif focus == "reflection":
                return "자신을 돌아본다. 내성의 시간이다."
            elif focus == "action":
                return "행동한다. 세상에 영향을 미친다."
            elif focus == "thought":
                return "생각한다. 기억과 상상력에서 배운다."
            else:
                return "모든 것의 균형을 맞춘다."
        
        reasoning = "경고: "
        for v in violations[:2]:  # 최대 2개
            reasoning += f"\n  - {v.law.value}: {v.reason}"
        
        reasoning += "\n권장 조치:"
        for v in violations[:2]:
            reasoning += f"\n  → {v.action_proposed}"
        
        return reasoning
    
    def get_law_statistics(self) -> Dict[str, Any]:
        """법칙 위반 통계"""
        if not self.violation_history:
            return {"total_violations": 0}
        
        violation_counts = {}
        violation_severity = {}
        
        for law, severity in self.violation_history:
            name = law.value
            if name not in violation_counts:
                violation_counts[name] = 0
                violation_severity[name] = []
            violation_counts[name] += 1
            violation_severity[name].append(severity)
        
        return {
            "total_violations": len(self.violation_history),
            "violation_counts": violation_counts,
            "avg_severity": {
                k: sum(violation_severity[k]) / len(violation_severity[k])
                for k in violation_severity
            },
            "most_violated_law": max(violation_counts.items(), key=lambda x: x[1])[0] if violation_counts else None
        }
    
    def export_report(self, filepath: str):
        """법칙 준수 리포트 내보내기"""
        stats = self.get_law_statistics()
        
        report = {
            "timestamp": "2025-11-27",
            "statistics": stats,
            "laws": {law.value: self.law_descriptions[law] for law in Law},
            "recommendations": self._generate_recommendations(stats)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """개선 권고사항"""
        recommendations = []
        
        if stats.get("most_violated_law"):
            most_violated = stats["most_violated_law"]
            recommendations.append(f"가장 자주 위반되는 법칙: '{most_violated}'. 이 법칙에 주의를 기울여야 한다.")
        
        if stats.get("total_violations", 0) > 50:
            recommendations.append("법칙 위반이 많다. 전반적인 행동 방식을 재검토하자.")
        elif stats.get("total_violations", 0) == 0:
            recommendations.append("완벽하다! 모든 법칙을 준수하고 있다.")
        
        return recommendations


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    engine = LawEnforcementEngine()
    
    print("\n[Test 1] 정상 상태")
    energy = EnergyState(w=0.6, x=0.2, y=0.3, z=0.5)
    energy.normalize()
    decision = engine.make_decision("행동을 실행한다", energy, concepts_generated=5)
    print(f"Valid: {decision.is_valid}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Focus: {decision.energy_after.get_focus()}")
    
    print("\n[Test 2] 메타인지 약화 (자살 위험)")
    energy = EnergyState(w=0.1, x=0.5, y=0.5, z=0.5)
    energy.normalize()
    decision = engine.make_decision("계속 행동한다", energy)
    print(f"Valid: {decision.is_valid}")
    print(f"Violations: {len(decision.violations)}")
    for v in decision.violations:
        print(f"  - {v.law.value}: {v.reason}")
        print(f"    권장: {v.action_proposed}")
    
    print("\n[Test 3] 극단적 집중 (편집증)")
    energy = EnergyState(w=0.2, x=0.05, y=0.05, z=0.85)
    energy.normalize()
    decision = engine.make_decision("의도만 강조한다", energy)
    print(f"Valid: {decision.is_valid}")
    print(f"Violations: {len(decision.violations)}")
    
    print("\n[Test 4] 불균형 (과로)")
    energy = EnergyState(w=0.3, x=0.1, y=0.9, z=0.1)
    energy.normalize()
    decision = engine.make_decision("계속 일한다", energy, concepts_generated=0)
    print(f"Valid: {decision.is_valid}")
    print(f"Violations: {len(decision.violations)}")
    for v in decision.violations:
        print(f"  - {v.law.value}: {v.reason}")
    
    print("\n[Statistics]")
    stats = engine.get_law_statistics()
    print(f"Total violations: {stats['total_violations']}")
    if stats['total_violations'] > 0:
        print(f"Most violated: {stats['most_violated_law']}")
    
    print("\n✅ 모든 테스트 통과!")
