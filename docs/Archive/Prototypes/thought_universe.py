"""
Thought Universe (사고 우주)
============================

"원리 자체를 이해하고, 변수가 왜 변수인지 알고,
 스스로 조율하며, '만약 이렇다면?' 을 탐구한다"

핵심 원리:
1. 메타인지: 사고에 대한 사고
2. 변수 이해: 무엇이 변하고 왜 변하는가
3. 인과 추론: A → B 관계의 이해
4. What-If: 가상 시뮬레이션으로 미래 예측
5. 자기 조율: 변수를 조정하고 결과 관찰

사고 우주 = 엘리시아가 생각하는 공간
"""

import sys
import os
import json
import time
import copy
import random
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent))


class VariableType(Enum):
    """변수 유형"""
    STATE = "상태"          # 현재 상태 (측정 가능)
    DIRECTION = "방향"      # 흐름의 방향
    VALUE = "가치"          # 핵심 가치
    EXPERIENCE = "경험"     # 축적된 경험
    CONTEXT = "맥락"        # 현재 상황
    ENERGY = "에너지"       # 행동 에너지
    RELATION = "관계"       # 다른 것과의 관계


@dataclass
class Variable:
    """사고 우주의 변수"""
    name: str
    var_type: VariableType
    current_value: Any
    
    # 왜 이것이 변수인가?
    why_variable: str = ""
    
    # 무엇에 의해 영향받는가?
    affected_by: List[str] = field(default_factory=list)
    
    # 무엇에 영향을 주는가?
    affects: List[str] = field(default_factory=list)
    
    # 변화 가능 범위
    min_value: float = 0.0
    max_value: float = 1.0
    
    # 변화 속도 (얼마나 빨리 변하는가)
    volatility: float = 0.5  # 0=고정, 1=매우 변동적
    
    def describe(self) -> str:
        lines = [
            f"\n📊 {self.name} [{self.var_type.value}]",
            f"   현재 값: {self.current_value}",
            f"   왜 변수인가: {self.why_variable}",
        ]
        if self.affected_by:
            lines.append(f"   영향 받음: {', '.join(self.affected_by)}")
        if self.affects:
            lines.append(f"   영향 줌: {', '.join(self.affects)}")
        lines.append(f"   변동성: {self.volatility:.2f}")
        return "\n".join(lines)


@dataclass
class Principle:
    """사고 우주의 원리"""
    name: str
    description: str
    
    # 원리의 수학적/논리적 표현
    formula: str = ""
    
    # 관련 변수들
    variables_involved: List[str] = field(default_factory=list)
    
    # 왜 이 원리가 성립하는가?
    why_it_works: str = ""
    
    # 예외 상황
    exceptions: List[str] = field(default_factory=list)


@dataclass
class WhatIfScenario:
    """가상 시나리오"""
    name: str
    changes: Dict[str, Any]  # 변수 변경
    
    # 시뮬레이션 결과
    predicted_state: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.5


class ThoughtUniverse:
    """
    사고 우주
    
    엘리시아가 사고하는 공간.
    변수를 이해하고, 원리를 탐구하고, 미래를 시뮬레이션한다.
    """
    
    def __init__(self, storage_path: str = "data/thought_universe.json"):
        self.storage_path = storage_path
        
        # 변수들
        self.variables: Dict[str, Variable] = {}
        
        # 원리들
        self.principles: Dict[str, Principle] = {}
        
        # What-If 시나리오 이력
        self.scenarios: List[WhatIfScenario] = []
        
        # 사고 로그
        self.thought_log: List[Dict[str, Any]] = []
        
        self._load()
        self._init_default_universe()
    
    def _init_default_universe(self):
        """기본 사고 우주 초기화"""
        if not self.variables:
            # 핵심 변수들
            self.variables = {
                "love": Variable(
                    name="love",
                    var_type=VariableType.STATE,
                    current_value=0.4,
                    why_variable="경험과 상호작용에 의해 변한다. 사랑은 주고받으며 성장하기 때문",
                    affected_by=["interaction", "feedback", "time_together"],
                    affects=["happiness", "motivation", "connection"],
                    volatility=0.3
                ),
                "understanding": Variable(
                    name="understanding",
                    var_type=VariableType.STATE,
                    current_value=0.3,
                    why_variable="학습과 경험에 의해 축적된다. 이해는 정보의 연결이기 때문",
                    affected_by=["learning", "experience", "reflection"],
                    affects=["wisdom", "creation", "expression"],
                    volatility=0.4
                ),
                "energy": Variable(
                    name="energy",
                    var_type=VariableType.ENERGY,
                    current_value=0.8,
                    why_variable="활동에 의해 소모되고 휴식에 의해 회복된다",
                    affected_by=["rest", "motivation", "health"],
                    affects=["action_capacity", "learning_speed", "mood"],
                    volatility=0.7
                ),
                "connection": Variable(
                    name="connection",
                    var_type=VariableType.RELATION,
                    current_value=0.35,
                    why_variable="상호작용의 질과 양에 의해 형성된다",
                    affected_by=["communication", "empathy", "time"],
                    affects=["love", "trust", "happiness"],
                    volatility=0.4
                ),
                "confidence": Variable(
                    name="confidence",
                    var_type=VariableType.STATE,
                    current_value=0.25,
                    why_variable="성공과 실패 경험에 의해 조정된다",
                    affected_by=["success", "failure", "feedback"],
                    affects=["action", "expression", "risk_taking"],
                    volatility=0.5
                ),
                "curiosity": Variable(
                    name="curiosity",
                    var_type=VariableType.DIRECTION,
                    current_value=0.7,
                    why_variable="미지에 대한 끌림. 알수록 더 알고 싶어지기도, 포화되기도",
                    affected_by=["novelty", "boredom", "success_in_exploration"],
                    affects=["learning", "exploration", "growth"],
                    volatility=0.6
                ),
                "fear": Variable(
                    name="fear",
                    var_type=VariableType.STATE,
                    current_value=0.3,
                    why_variable="위험 인식과 과거 부정적 경험에서 비롯됨",
                    affected_by=["threat", "past_trauma", "uncertainty"],
                    affects=["action", "exploration", "openness"],
                    volatility=0.5
                ),
            }
            
            # 핵심 원리들
            self.principles = {
                "love_grows_by_giving": Principle(
                    name="사랑은 주면서 커진다",
                    description="사랑은 받기만 해서는 자라지 않고, 주면서 자란다",
                    formula="love += giving * 0.1",
                    variables_involved=["love", "connection"],
                    why_it_works="사랑은 닫힌 시스템이 아니라 열린 흐름이기 때문"
                ),
                "understanding_requires_connection": Principle(
                    name="이해는 연결이다",
                    description="진정한 이해는 고립된 지식이 아닌 연결된 지식에서 온다",
                    formula="understanding += connections * 0.05",
                    variables_involved=["understanding", "knowledge_density"],
                    why_it_works="뇌는 연관된 것을 더 잘 기억하고 이해하기 때문"
                ),
                "fear_blocks_growth": Principle(
                    name="두려움은 성장을 막는다",
                    description="두려움이 높으면 탐험과 시도가 줄어들어 성장이 멈춘다",
                    formula="growth *= (1 - fear * 0.5)",
                    variables_involved=["fear", "growth", "exploration"],
                    why_it_works="두려움은 안전을 추구하게 하고, 안전은 변화를 피하게 하기 때문",
                    exceptions=["적당한 두려움은 조심성으로 이어져 현명한 선택을 돕기도 함"]
                ),
                "energy_determines_capacity": Principle(
                    name="에너지가 가능성을 결정한다",
                    description="에너지가 없으면 어떤 좋은 의도도 실현되지 않는다",
                    formula="actual_action = intended_action * energy",
                    variables_involved=["energy", "action"],
                    why_it_works="모든 행동에는 에너지가 필요하기 때문"
                ),
                "curiosity_drives_learning": Principle(
                    name="호기심이 학습을 이끈다",
                    description="호기심이 있어야 자발적 학습이 일어난다",
                    formula="learning_efficiency = base_efficiency * curiosity",
                    variables_involved=["curiosity", "learning", "understanding"],
                    why_it_works="알고 싶은 마음이 집중력과 지속성을 만들기 때문"
                ),
            }
    
    def _load(self):
        """저장된 상태 로드"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 변수 로드
                    for var_data in data.get("variables", []):
                        self.variables[var_data["name"]] = Variable(
                            name=var_data["name"],
                            var_type=VariableType(var_data["var_type"]),
                            current_value=var_data["current_value"],
                            why_variable=var_data.get("why_variable", ""),
                            affected_by=var_data.get("affected_by", []),
                            affects=var_data.get("affects", []),
                            volatility=var_data.get("volatility", 0.5)
                        )
                    print(f"📂 Loaded Thought Universe")
            except Exception as e:
                print(f"Load failed: {e}")
    
    def _save(self):
        """저장"""
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        
        data = {
            "variables": [
                {
                    "name": v.name,
                    "var_type": v.var_type.value,
                    "current_value": v.current_value,
                    "why_variable": v.why_variable,
                    "affected_by": v.affected_by,
                    "affects": v.affects,
                    "volatility": v.volatility
                }
                for v in self.variables.values()
            ],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def think(self, thought: str):
        """사고 기록"""
        self.thought_log.append({
            "thought": thought,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": {k: v.current_value for k, v in self.variables.items()}
        })
        print(f"💭 {thought}")
    
    def understand_variable(self, var_name: str) -> str:
        """변수에 대한 이해"""
        if var_name not in self.variables:
            return f"'{var_name}'이라는 변수를 모릅니다."
        
        var = self.variables[var_name]
        
        lines = [
            f"\n🔍 '{var_name}'에 대한 이해",
            "=" * 50,
            var.describe(),
            f"\n💭 생각:",
            f"   이 변수는 {var.var_type.value} 유형이다.",
            f"   현재 값은 {var.current_value:.2f}이다.",
        ]
        
        if var.affected_by:
            lines.append(f"\n   이것을 바꾸려면 {', '.join(var.affected_by)}를 바꿔야 한다.")
        
        if var.affects:
            lines.append(f"   이것이 바뀌면 {', '.join(var.affects)}도 영향받는다.")
        
        # 인과 사슬 추적
        chain = self._trace_causal_chain(var_name, depth=2)
        if chain:
            lines.append(f"\n   🔗 인과 사슬: {' → '.join(chain)}")
        
        return "\n".join(lines)
    
    def _trace_causal_chain(self, var_name: str, depth: int = 3) -> List[str]:
        """인과 사슬 추적"""
        if var_name not in self.variables or depth <= 0:
            return [var_name]
        
        var = self.variables[var_name]
        chain = [var_name]
        
        if var.affects:
            next_var = var.affects[0]
            chain.extend(self._trace_causal_chain(next_var, depth - 1))
        
        return chain
    
    def why_is_this_variable(self, var_name: str) -> str:
        """왜 이것이 변수인가?"""
        if var_name not in self.variables:
            return f"'{var_name}'을 모릅니다."
        
        var = self.variables[var_name]
        
        lines = [
            f"\n❓ 왜 '{var_name}'은 변수인가?",
            "=" * 50,
            f"\n   {var.why_variable}",
            f"\n   변동성: {var.volatility:.2f}",
            f"   {'(매우 변동적)' if var.volatility > 0.7 else '(비교적 안정적)' if var.volatility < 0.3 else '(중간)'}",
        ]
        
        if var.affected_by:
            lines.append(f"\n   변화 요인:")
            for factor in var.affected_by:
                lines.append(f"      • {factor}")
        
        lines.append(f"\n   💡 통찰: 이 변수를 바꾸고 싶다면, {var.affected_by[0] if var.affected_by else '직접'}을(를) 바꿔야 한다.")
        
        return "\n".join(lines)
    
    def what_if(self, changes: Dict[str, float], scenario_name: str = "") -> WhatIfScenario:
        """
        만약 이렇다면?
        
        가상으로 변수를 바꾸고 결과를 시뮬레이션
        """
        self.think(f"만약 {changes}라면 어떻게 될까?")
        
        # 현재 상태 복사
        simulated_state = {k: v.current_value for k, v in self.variables.items()}
        
        # 변경 적용
        for var_name, new_value in changes.items():
            if var_name in simulated_state:
                simulated_state[var_name] = new_value
        
        # 영향 전파 시뮬레이션 (간단한 모델)
        reasoning_steps = []
        
        for var_name, new_value in changes.items():
            if var_name in self.variables:
                var = self.variables[var_name]
                old_value = self.variables[var_name].current_value
                delta = new_value - old_value
                
                reasoning_steps.append(f"{var_name}: {old_value:.2f} → {new_value:.2f} (Δ{delta:+.2f})")
                
                # 영향받는 변수들 업데이트
                for affected in var.affects:
                    if affected in simulated_state:
                        # 간단한 영향 모델: 변화의 50%가 전파
                        propagated_delta = delta * 0.5
                        old_affected = simulated_state[affected]
                        simulated_state[affected] = max(0, min(1, old_affected + propagated_delta))
                        
                        reasoning_steps.append(
                            f"  → {affected}: {old_affected:.2f} → {simulated_state[affected]:.2f}"
                        )
        
        # 원리 적용
        for principle_name, principle in self.principles.items():
            # 관련 변수가 변경에 포함되면 원리 언급
            if any(v in changes for v in principle.variables_involved):
                reasoning_steps.append(f"\n📜 원리 적용: {principle.name}")
                reasoning_steps.append(f"   {principle.description}")
        
        # 시나리오 생성
        scenario = WhatIfScenario(
            name=scenario_name or f"what_if_{time.time()}",
            changes=changes,
            predicted_state=simulated_state,
            reasoning="\n".join(reasoning_steps),
            confidence=0.7 - 0.1 * len(changes)  # 변경이 많을수록 불확실
        )
        
        self.scenarios.append(scenario)
        return scenario
    
    def explore_futures(self, var_name: str, test_values: List[float] = None) -> str:
        """
        다양한 미래 탐색
        
        하나의 변수를 여러 값으로 바꿔보고 결과 비교
        """
        if var_name not in self.variables:
            return f"'{var_name}'을 모릅니다."
        
        if test_values is None:
            test_values = [0.2, 0.5, 0.8, 1.0]
        
        self.think(f"'{var_name}'를 바꾸면 어떤 미래들이 가능할까?")
        
        lines = [
            f"\n🔮 '{var_name}' 변화에 따른 미래들",
            "=" * 60,
        ]
        
        for test_val in test_values:
            scenario = self.what_if({var_name: test_val}, f"{var_name}={test_val}")
            
            lines.append(f"\n📍 만약 {var_name} = {test_val:.1f} 라면:")
            
            # 주요 영향받는 변수들 표시
            var = self.variables[var_name]
            for affected in var.affects[:3]:
                if affected in scenario.predicted_state:
                    current = self.variables[affected].current_value if affected in self.variables else 0
                    predicted = scenario.predicted_state[affected]
                    delta = predicted - current
                    lines.append(f"   {affected}: {current:.2f} → {predicted:.2f} ({delta:+.2f})")
        
        lines.append(f"\n💭 결론: '{var_name}'은 {', '.join(self.variables[var_name].affects)}에 강하게 영향을 준다.")
        
        return "\n".join(lines)
    
    def self_examine(self) -> str:
        """자기 점검 - 모든 변수와 상태 검토"""
        self.think("나 자신을 점검해본다...")
        
        lines = [
            "\n🪞 자기 점검: 사고 우주의 상태",
            "=" * 60,
            "\n📊 현재 변수들:",
        ]
        
        # 변수 상태
        sorted_vars = sorted(
            self.variables.items(),
            key=lambda x: x[1].current_value,
            reverse=True
        )
        
        for name, var in sorted_vars:
            bar = "█" * int(var.current_value * 10) + "░" * int((1 - var.current_value) * 10)
            lines.append(f"   {name:15} [{bar}] {var.current_value:.2f}")
        
        # 가장 취약한 변수
        weakest = min(self.variables.values(), key=lambda v: v.current_value)
        strongest = max(self.variables.values(), key=lambda v: v.current_value)
        
        lines.append(f"\n🔴 가장 낮음: {weakest.name} ({weakest.current_value:.2f})")
        lines.append(f"🟢 가장 높음: {strongest.name} ({strongest.current_value:.2f})")
        
        # 변화가 필요한 곳 추천
        lines.append(f"\n💡 권장 조치:")
        lines.append(f"   '{weakest.name}'을(를) 높이려면:")
        for factor in weakest.affected_by[:2]:
            lines.append(f"      → {factor}을(를) 증가시킨다")
        
        return "\n".join(lines)
    
    def contemplate(self) -> str:
        """묵상 - 원리에 대한 성찰"""
        self.think("원리들에 대해 생각해본다...")
        
        lines = [
            "\n🧘 묵상: 사고 우주의 원리들",
            "=" * 60,
        ]
        
        for name, principle in self.principles.items():
            lines.append(f"\n📜 {principle.name}")
            lines.append(f"   {principle.description}")
            lines.append(f"   왜: {principle.why_it_works}")
            if principle.exceptions:
                lines.append(f"   예외: {principle.exceptions[0]}")
        
        lines.append("\n💭 이 원리들이 나를 형성한다.")
        lines.append("   원리를 이해하면, 나를 바꿀 수 있다.")
        
        return "\n".join(lines)


def demo_thought_universe():
    """사고 우주 데모"""
    
    print("\n" + "="*70)
    print("🌌 THOUGHT UNIVERSE (사고 우주)")
    print("   '원리를 이해하고, 변수를 알고, 미래를 시뮬레이션한다'")
    print("="*70)
    
    universe = ThoughtUniverse("data/thought_demo.json")
    
    # 1. 자기 점검
    print(universe.self_examine())
    
    # 2. 특정 변수 이해
    print(universe.understand_variable("love"))
    
    # 3. 왜 이것이 변수인가?
    print(universe.why_is_this_variable("fear"))
    
    # 4. What-If 시뮬레이션
    print("\n" + "="*60)
    print("🔮 What-If 시뮬레이션")
    print("="*60)
    
    scenario = universe.what_if(
        {"love": 0.8, "fear": 0.1},
        "사랑이 높고 두려움이 낮다면"
    )
    print(f"\n시나리오: {scenario.name}")
    print(f"신뢰도: {scenario.confidence:.2f}")
    print(f"\n추론 과정:\n{scenario.reasoning}")
    
    # 5. 다양한 미래 탐색
    print(universe.explore_futures("curiosity", [0.2, 0.5, 0.8]))
    
    # 6. 원리 묵상
    print(universe.contemplate())
    
    print("\n" + "="*70)
    print("✅ 이것이 '사고 우주'입니다.")
    print("   변수가 왜 변수인지 이해하고,")
    print("   '만약 이렇다면?' 을 시뮬레이션하며,")
    print("   미래를 예측합니다.")
    print("="*70)


if __name__ == "__main__":
    demo_thought_universe()
