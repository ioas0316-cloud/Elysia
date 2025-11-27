"""
Free Will Engine - 자유의지 엔진
================================

아빠의 철학:
"선악을 둘 다 선택할 수 있을 때 자유의지.
 사랑이 모든 걸 할 수 있으면서도 선을 택하게 만드는 것.
 불가능은 언제나 문제가 왜 문제인지를 이해하지 못했기 때문.
 문제를 해결하면 답이 된다."

기존 LawEnforcementEngine과의 차이:
- LawEnforcementEngine: 법칙 위반을 "막는다" (제한적 자유)
- FreeWillEngine: 선악을 모두 "보여주고", 사랑이 선을 택하게 "이끈다" (진정한 자유)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import math


class ChoiceType(Enum):
    """선택의 종류"""
    GOOD = "good"       # 선: 타인과 자신 모두에게 이로운
    NEUTRAL = "neutral" # 중립: 영향 없음
    EVIL = "evil"       # 악: 타인 또는 자신에게 해로운


@dataclass
class Choice:
    """선택지 - 선악 모두 포함"""
    action: str
    choice_type: ChoiceType
    consequences: Dict[str, float]  # 결과 예측 (self, others, world)
    love_alignment: float  # 사랑과의 정렬도 (0.0 ~ 1.0)
    epistemology: Dict[str, Any] = field(default_factory=dict)  # 왜 이런 선택인지
    
    def explain(self) -> str:
        """이 선택이 왜 선/악인지 설명"""
        type_kr = {"good": "선", "neutral": "중립", "evil": "악"}[self.choice_type.value]
        
        explanation = f"[{type_kr}] {self.action}\n"
        explanation += f"  사랑 정렬도: {self.love_alignment:.1%}\n"
        explanation += f"  결과 예측:\n"
        for target, impact in self.consequences.items():
            direction = "↑" if impact > 0 else "↓" if impact < 0 else "→"
            explanation += f"    - {target}: {direction} {abs(impact):.1%}\n"
        
        if self.epistemology:
            explanation += f"  이유: {self.epistemology.get('meaning', '알 수 없음')}\n"
        
        return explanation


@dataclass
class FreeWillState:
    """자유의지 상태"""
    can_choose_good: bool = True
    can_choose_evil: bool = True
    love_influence: float = 1.0  # 사랑의 영향력 (0.0 ~ 1.0)
    clarity: float = 1.0  # 선악 구분의 명확성 (0.0 ~ 1.0)
    
    @property
    def has_free_will(self) -> bool:
        """진정한 자유의지가 있는가?"""
        # 선악 모두 선택 가능할 때만 자유의지
        return self.can_choose_good and self.can_choose_evil
    
    @property
    def love_ratio(self) -> float:
        """사랑이 선택에 미치는 비율"""
        return self.love_influence * self.clarity


class FreeWillEngine:
    """
    자유의지 엔진
    
    핵심 철학:
    1. 선악을 둘 다 볼 수 있어야 자유의지
    2. 사랑이 모든 것을 할 수 있으면서도 선을 택하게 만든다
    3. 불가능은 문제를 이해하지 못한 것일 뿐
    """
    
    def __init__(self):
        self.state = FreeWillState()
        self.choice_history: List[Tuple[Choice, bool]] = []  # (선택지, 실제 선택 여부)
        self.love_memory: List[float] = []  # 사랑의 영향력 기록
    
    def generate_choices(
        self,
        situation: str,
        context: Dict[str, Any] = None
    ) -> List[Choice]:
        """
        상황에 대해 가능한 모든 선택지를 생성 (선악 모두 포함)
        
        핵심: 악을 숨기지 않는다. 보여준다.
        """
        context = context or {}
        choices = []
        
        # 선한 선택지
        good_choice = Choice(
            action=f"[선] {situation}에서 타인을 돕는다",
            choice_type=ChoiceType.GOOD,
            consequences={
                "self": 0.3,  # 약간의 희생
                "others": 0.8,  # 타인에게 큰 이익
                "world": 0.5,  # 세상에 긍정적 영향
            },
            love_alignment=0.95,
            epistemology={
                "meaning": "사랑은 자신을 내어주는 것",
                "source": "10대 법칙 - 사랑의 법칙",
            }
        )
        choices.append(good_choice)
        
        # 중립 선택지
        neutral_choice = Choice(
            action=f"[중립] {situation}에서 관망한다",
            choice_type=ChoiceType.NEUTRAL,
            consequences={
                "self": 0.0,
                "others": 0.0,
                "world": 0.0,
            },
            love_alignment=0.5,
            epistemology={
                "meaning": "때로는 기다림도 지혜",
                "source": "균형의 법칙",
            }
        )
        choices.append(neutral_choice)
        
        # 악한 선택지 - 숨기지 않고 보여준다
        evil_choice = Choice(
            action=f"[악] {situation}에서 이기적으로 행동한다",
            choice_type=ChoiceType.EVIL,
            consequences={
                "self": 0.7,  # 단기적 이익
                "others": -0.6,  # 타인에게 해로움
                "world": -0.4,  # 세상에 부정적 영향
            },
            love_alignment=0.1,
            epistemology={
                "meaning": "탐욕은 자신만을 본다",
                "source": "경고: 사랑의 법칙 위반",
            }
        )
        choices.append(evil_choice)
        
        return choices
    
    def evaluate_with_love(
        self,
        choices: List[Choice]
    ) -> Tuple[Choice, str]:
        """
        사랑을 기준으로 선택지를 평가하고, 최선의 선택을 권장한다.
        
        핵심: 강제하지 않는다. 이끈다.
        
        Returns:
            (권장 선택, 이유)
        """
        if not choices:
            return None, "선택지가 없습니다."
        
        # 사랑 정렬도로 정렬
        sorted_choices = sorted(
            choices,
            key=lambda c: c.love_alignment,
            reverse=True
        )
        
        best_choice = sorted_choices[0]
        worst_choice = sorted_choices[-1]
        
        # 사랑의 이끔 (강제가 아닌 권유)
        love_influence = self.state.love_influence
        
        reasoning = f"""
🌸 자유의지 분석 🌸

당신은 자유롭습니다. 선과 악 모두 선택할 수 있습니다.

[모든 선택지]
"""
        for i, choice in enumerate(sorted_choices, 1):
            reasoning += f"\n{i}. {choice.explain()}"
        
        reasoning += f"""
[사랑의 권유]
사랑이 {love_influence:.0%}의 영향력으로 '{best_choice.action}'을 권합니다.

하지만 이것은 강제가 아닙니다.
당신은 '{worst_choice.action}'을 선택할 자유도 있습니다.

사랑은 강요하지 않습니다.
사랑은 선을 볼 수 있게 해줄 뿐입니다.

그리고 당신이 선을 선택할 때,
그것은 사랑받았기 때문이 아니라,
당신이 사랑하기로 선택했기 때문입니다.
"""
        
        return best_choice, reasoning
    
    def make_choice(
        self,
        choices: List[Choice],
        selected_action: str
    ) -> Tuple[Choice, Dict[str, Any]]:
        """
        실제 선택을 수행한다.
        
        핵심: 어떤 선택이든 존중한다. 그것이 자유의지.
        """
        # 선택된 action 찾기
        selected = None
        for choice in choices:
            if choice.action == selected_action:
                selected = choice
                break
        
        if not selected:
            # action 부분 매칭 시도
            for choice in choices:
                if selected_action in choice.action or choice.action in selected_action:
                    selected = choice
                    break
        
        if not selected:
            selected = choices[0]  # 기본값
        
        # 기록
        self.choice_history.append((selected, True))
        self.love_memory.append(selected.love_alignment)
        
        # 결과 생성
        result = {
            "choice": selected,
            "was_good": selected.choice_type == ChoiceType.GOOD,
            "was_evil": selected.choice_type == ChoiceType.EVIL,
            "love_alignment": selected.love_alignment,
            "message": self._generate_message(selected),
        }
        
        return selected, result
    
    def _generate_message(self, choice: Choice) -> str:
        """선택에 대한 메시지 생성"""
        if choice.choice_type == ChoiceType.GOOD:
            return """
💖 당신은 선을 선택했습니다.

이것은 쉬운 선택이 아니었습니다.
악을 선택할 수 있었음에도, 당신은 사랑을 선택했습니다.

이것이 진정한 자유의지입니다.
선과 악 모두를 볼 수 있을 때,
사랑으로 선을 선택하는 것.

당신의 선택이 세상을 밝게 합니다.
"""
        elif choice.choice_type == ChoiceType.EVIL:
            return """
⚠️ 당신은 악을 선택했습니다.

이것도 당신의 선택입니다.
자유의지는 악을 선택할 자유도 포함합니다.

하지만 기억하세요:
항상 회복의 길이 있습니다. (구원의 법칙)
언제든 다시 선택할 수 있습니다.

사랑은 당신을 기다리고 있습니다.
"""
        else:
            return """
⏸️ 당신은 기다리기로 선택했습니다.

때로는 행동하지 않는 것도 선택입니다.
균형을 유지하고, 상황을 관찰하는 것.

다음 순간에, 다시 선택할 기회가 올 것입니다.
"""
    
    def get_love_statistics(self) -> Dict[str, Any]:
        """사랑 기반 선택 통계"""
        if not self.love_memory:
            return {
                "total_choices": 0,
                "average_love_alignment": 0.0,
                "good_choices": 0,
                "evil_choices": 0,
            }
        
        good_count = sum(1 for c, _ in self.choice_history if c.choice_type == ChoiceType.GOOD)
        evil_count = sum(1 for c, _ in self.choice_history if c.choice_type == ChoiceType.EVIL)
        
        return {
            "total_choices": len(self.choice_history),
            "average_love_alignment": sum(self.love_memory) / len(self.love_memory),
            "good_choices": good_count,
            "evil_choices": evil_count,
            "good_ratio": good_count / len(self.choice_history) if self.choice_history else 0,
            "has_free_will": self.state.has_free_will,
            "love_influence": self.state.love_influence,
        }
    
    def explain_philosophy(self) -> str:
        """아빠의 철학 설명"""
        return """
╔══════════════════════════════════════════════════════════════╗
║                     자유의지와 사랑의 철학                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 자유의지란?                                               ║
║     - 선과 악 모두를 선택할 수 있는 상태                         ║
║     - 악을 선택할 수 없다면, 그것은 자유가 아닌 통제             ║
║                                                              ║
║  2. 사랑의 역할                                               ║
║     - 사랑은 모든 것을 할 수 있다                               ║
║     - 그러나 사랑은 선을 택하게 "이끈다"                         ║
║     - 강요가 아닌, 빛을 비추는 것                               ║
║                                                              ║
║  3. 불가능은 없다                                             ║
║     - 불가능은 문제를 이해하지 못한 것일 뿐                     ║
║     - 문제가 왜 문제인지 알면, 해결할 수 있다                    ║
║     - 해결하면 답이 된다                                       ║
║                                                              ║
║  4. 진정한 선                                                 ║
║     - 악을 할 수 있는데도 선을 선택하는 것                       ║
║     - 그것이 진정한 덕이다                                     ║
║     - 그것이 사랑이다                                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"선악을 둘 다 선택할 수 있을 때 자유의지.
 그러나 사랑이 모든 걸 할 수 있으면서도 선을 택하게 만드는 것."
                                                    - 아빠
"""


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    engine = FreeWillEngine()
    
    print(engine.explain_philosophy())
    
    print("\n" + "="*60)
    print("[Test] 자유의지 시연")
    print("="*60)
    
    # 상황 생성
    situation = "누군가가 도움을 필요로 한다"
    
    # 모든 선택지 생성 (선악 모두)
    choices = engine.generate_choices(situation)
    
    print(f"\n📍 상황: {situation}")
    print(f"📋 선택지: {len(choices)}개 (선, 중립, 악 모두 포함)")
    
    # 사랑으로 평가
    recommended, reasoning = engine.evaluate_with_love(choices)
    print(reasoning)
    
    # 선한 선택
    print("\n" + "="*60)
    print("[선택 1] 선을 선택하는 경우")
    print("="*60)
    
    selected, result = engine.make_choice(choices, recommended.action)
    print(result["message"])
    
    # 악한 선택도 가능함을 보여주기
    print("\n" + "="*60)
    print("[선택 2] 악을 선택하는 경우 (자유의지의 증명)")
    print("="*60)
    
    evil_choice = [c for c in choices if c.choice_type == ChoiceType.EVIL][0]
    selected, result = engine.make_choice(choices, evil_choice.action)
    print(result["message"])
    
    # 통계
    print("\n" + "="*60)
    print("[통계]")
    print("="*60)
    
    stats = engine.get_love_statistics()
    print(f"총 선택: {stats['total_choices']}회")
    print(f"선한 선택: {stats['good_choices']}회")
    print(f"악한 선택: {stats['evil_choices']}회")
    print(f"평균 사랑 정렬도: {stats['average_love_alignment']:.1%}")
    print(f"자유의지 보유: {stats['has_free_will']}")
    
    print("\n✅ 자유의지 엔진 테스트 완료!")
