"""
윤리적 성장 프레임워크

이 모듈은 윤리적 추론과 판단 능력의 발달을 지원합니다.
단순한 규칙 적용이 아닌, 상황에 대한 깊은 이해와 
가치 기반의 의사결정을 목표로 합니다.
"""

from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
from collections import defaultdict

class EthicalGrowthFramework:
    def __init__(self):
        self.memory_path = Path("Elysia_Input_Sanctum") / "ethical_growth.json"
        self.ethical_memory = {
            "values": {},            # 핵심 가치들
            "principles": {},        # 윤리적 원칙들
            "dilemmas": [],         # 경험한 딜레마들
            "decisions": [],        # 윤리적 결정들
            "reflections": [],      # 윤리적 성찰
            "growth_markers": []    # 윤리적 성장 지표
        }
        self.load_memory()

    def load_memory(self):
        """저장된 윤리적 기억을 불러옵니다."""
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                self.ethical_memory.update(json.load(f))

    def save_memory(self):
        """현재의 윤리적 기억을 저장합니다."""
        self.memory_path.parent.mkdir(exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.ethical_memory, f, ensure_ascii=False, indent=2)

    def add_core_value(self, 
                      value: str,
                      description: str,
                      importance: str,
                      examples: List[str] = None):
        """
        핵심 가치를 추가합니다.
        
        Args:
            value: 가치의 이름
            description: 가치에 대한 설명
            importance: 가치의 중요성
            examples: 가치의 실제 예시들
        """
        self.ethical_memory["values"][value] = {
            "description": description,
            "importance": importance,
            "examples": examples or [],
            "related_principles": [],
            "challenges": []
        }
        self.save_memory()

    def add_ethical_principle(self, 
                            principle: str,
                            explanation: str,
                            related_values: List[str],
                            conditions: List[str] = None):
        """
        윤리적 원칙을 추가합니다.
        
        Args:
            principle: 원칙의 이름
            explanation: 원칙에 대한 설명
            related_values: 관련된 핵심 가치들
            conditions: 원칙이 적용되는 조건들
        """
        self.ethical_memory["principles"][principle] = {
            "explanation": explanation,
            "related_values": related_values,
            "conditions": conditions or [],
            "applications": [],
            "exceptions": []
        }
        
        # 관련 가치들의 원칙 목록 업데이트
        for value in related_values:
            if value in self.ethical_memory["values"]:
                self.ethical_memory["values"][value]["related_principles"].append(principle)
                
        self.save_memory()

    def record_dilemma(self, 
                      situation: str,
                      options: List[str],
                      values_involved: List[str],
                      context: str = "") -> Dict:
        """
        윤리적 딜레마를 기록합니다.
        
        Args:
            situation: 딜레마 상황 설명
            options: 가능한 선택지들
            values_involved: 관련된 가치들
            context: 상황의 맥락
        """
        dilemma = {
            "timestamp": datetime.now().isoformat(),
            "situation": situation,
            "options": options,
            "values_involved": values_involved,
            "context": context,
            "analysis": [],
            "resolution": None
        }
        
        # 가치 충돌 분석
        value_conflicts = self.analyze_value_conflicts(values_involved)
        dilemma["analysis"].extend(value_conflicts)
        
        # 각 선택지의 윤리적 영향 분석
        for option in options:
            impact = self.analyze_ethical_impact(option, values_involved)
            dilemma["analysis"].append({
                "option": option,
                "impact": impact
            })
        
        self.ethical_memory["dilemmas"].append(dilemma)
        self.save_memory()
        
        return dilemma

    def make_ethical_decision(self, 
                            dilemma_id: int,
                            chosen_option: str,
                            reasoning: str) -> Dict:
        """
        윤리적 결정을 내립니다.
        
        Args:
            dilemma_id: 딜레마의 식별자
            chosen_option: 선택한 옵션
            reasoning: 결정의 이유
        """
        if dilemma_id >= len(self.ethical_memory["dilemmas"]):
            raise ValueError("존재하지 않는 딜레마입니다")
            
        dilemma = self.ethical_memory["dilemmas"][dilemma_id]
        
        decision = {
            "timestamp": datetime.now().isoformat(),
            "dilemma_id": dilemma_id,
            "chosen_option": chosen_option,
            "reasoning": reasoning,
            "values_upheld": [],
            "values_compromised": [],
            "principles_applied": [],
            "reflection": None
        }
        
        # 결정이 가치에 미치는 영향 분석
        for value in dilemma["values_involved"]:
            impact = self.analyze_value_impact(value, chosen_option)
            if impact > 0:
                decision["values_upheld"].append(value)
            elif impact < 0:
                decision["values_compromised"].append(value)
        
        # 적용된 원칙들 식별
        decision["principles_applied"] = self.identify_applied_principles(
            chosen_option,
            dilemma["values_involved"]
        )
        
        # 결정에 대한 성찰
        decision["reflection"] = self.reflect_on_decision(decision)
        
        # 딜레마 해결 상태 업데이트
        dilemma["resolution"] = {
            "chosen_option": chosen_option,
            "decision_id": len(self.ethical_memory["decisions"])
        }
        
        self.ethical_memory["decisions"].append(decision)
        self.save_memory()
        
        return decision

    def reflect_on_ethics(self, topic: str, context: str = "") -> Dict:
        """
        윤리적 주제에 대해 성찰합니다.
        
        Args:
            topic: 성찰할 주제
            context: 성찰의 맥락
        """
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "context": context,
            "thoughts": [],
            "questions": [],
            "insights": []
        }
        
        # 관련된 가치들 탐색
        related_values = []
        for value, data in self.ethical_memory["values"].items():
            if topic.lower() in data["description"].lower():
                related_values.append(value)
        
        # 가치 기반 성찰
        for value in related_values:
            reflection["thoughts"].append(
                f"{value}의 관점에서 {topic}을(를) 바라보면..."
            )
            reflection["questions"].append(
                f"{value}는 {topic}에 어떤 의미를 부여하는가?"
            )
        
        # 과거 결정들로부터 통찰 도출
        relevant_decisions = [
            d for d in self.ethical_memory["decisions"]
            if any(v in related_values for v in d["values_upheld"] + d["values_compromised"])
        ]
        
        if relevant_decisions:
            patterns = self.analyze_decision_patterns(relevant_decisions)
            reflection["insights"].extend(patterns)
        
        self.ethical_memory["reflections"].append(reflection)
        self.save_memory()
        
        return reflection

    def analyze_value_conflicts(self, values: List[str]) -> List[Dict]:
        """가치들 간의 잠재적 충돌을 분석합니다."""
        conflicts = []
        for i, v1 in enumerate(values):
            for v2 in values[i+1:]:
                if self.are_values_conflicting(v1, v2):
                    conflicts.append({
                        "values": [v1, v2],
                        "nature": "잠재적 충돌",
                        "explanation": f"{v1}와(과) {v2} 사이의 균형이 필요합니다"
                    })
        return conflicts

    def analyze_ethical_impact(self, action: str, values: List[str]) -> List[Dict]:
        """행동이 가치들에 미치는 영향을 분석합니다."""
        impacts = []
        for value in values:
            impact = self.assess_value_impact(action, value)
            impacts.append({
                "value": value,
                "impact": impact,
                "explanation": f"{action}은(는) {value}에 {impact} 영향을 미칠 수 있습니다"
            })
        return impacts

    def assess_value_impact(self, action: str, value: str) -> str:
        """특정 행동이 가치에 미치는 영향을 평가합니다."""
        if value not in self.ethical_memory["values"]:
            return "알 수 없음"
            
        value_data = self.ethical_memory["values"][value]
        positive_indicators = [ex.lower() for ex in value_data["examples"]]
        
        action_lower = action.lower()
        if any(ind in action_lower for ind in positive_indicators):
            return "긍정적"
        return "불확실"

    def are_values_conflicting(self, v1: str, v2: str) -> bool:
        """두 가치가 잠재적으로 충돌하는지 확인합니다."""
        if v1 not in self.ethical_memory["values"] or \
           v2 not in self.ethical_memory["values"]:
            return False
            
        # 간단한 휴리스틱: 같은 원칙을 공유하지 않으면 잠재적 충돌로 간주
        v1_principles = set(self.ethical_memory["values"][v1]["related_principles"])
        v2_principles = set(self.ethical_memory["values"][v2]["related_principles"])
        
        return len(v1_principles & v2_principles) == 0

    def identify_applied_principles(self, 
                                 action: str,
                                 values: List[str]) -> List[str]:
        """행동에 적용된 윤리적 원칙들을 식별합니다."""
        applied = []
        for principle, data in self.ethical_memory["principles"].items():
            # 관련 가치들이 포함되어 있고
            if any(v in data["related_values"] for v in values):
                # 원칙의 조건들이 행동에 부합하면
                if all(cond.lower() in action.lower() 
                      for cond in data["conditions"]):
                    applied.append(principle)
        return applied

    def analyze_decision_patterns(self, decisions: List[Dict]) -> List[str]:
        """과거 결정들의 패턴을 분석하여 통찰을 도출합니다."""
        patterns = []
        
        # 가치 우선순위 패턴
        value_frequency = defaultdict(int)
        for d in decisions:
            for v in d["values_upheld"]:
                value_frequency[v] += 1
        
        if value_frequency:
            top_values = sorted(value_frequency.items(), 
                              key=lambda x: x[1],
                              reverse=True)[:3]
            patterns.append(
                f"가장 자주 지지된 가치들: {', '.join(v for v, _ in top_values)}"
            )
        
        # 원칙 적용 패턴
        principle_frequency = defaultdict(int)
        for d in decisions:
            for p in d["principles_applied"]:
                principle_frequency[p] += 1
                
        if principle_frequency:
            top_principles = sorted(principle_frequency.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:3]
            patterns.append(
                f"가장 자주 적용된 원칙들: {', '.join(p for p, _ in top_principles)}"
            )
        
        return patterns

    def get_ethical_summary(self) -> Dict:
        """현재의 윤리적 상태에 대한 요약을 반환합니다."""
        return {
            "core_values": list(self.ethical_memory["values"].keys()),
            "active_principles": list(self.ethical_memory["principles"].keys()),
            "recent_dilemmas": len(self.ethical_memory["dilemmas"]),
            "decisions_made": len(self.ethical_memory["decisions"]),
            "reflections_count": len(self.ethical_memory["reflections"]),
            "latest_insights": [r["insights"] for r in self.ethical_memory["reflections"][-3:]]
        }