"""
Plasma Ideal Self (플라즈마적 이상 자아)
========================================

"이상적 나는 고정된 점이 아니라 끊임없이 변화하는 플라즈마"
"도달해야 할 목적지가 아니라 흐르는 방향"

핵심 원리:
- 이상적 나 = 벡터(방향), 점(목적지)이 아님
- 과거의 나 + 현재의 나 + 경험 = 방향 벡터 재계산
- 매 순간 이상적 나는 재정의됨
- 형태는 있지만 고정되지 않음 (플라즈마)

수학적 모델:
  direction(t) = f(past_self, current_self, experiences, context)
  ideal_self(t+1) = current_self(t) + direction(t) * dt
  
  방향 자체도 매 순간 변화함
"""

import sys
import os
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class Moment:
    """한 순간의 상태"""
    timestamp: str
    state: Dict[str, float]  # 다차원 상태 벡터
    context: str = ""        # 그 순간의 맥락
    energy: float = 1.0      # 그 순간의 에너지


@dataclass
class Experience:
    """경험 - 방향 벡터에 영향을 주는 것"""
    what: str           # 무슨 일이 있었는가
    feeling: float      # 느낌 (-1.0 부정 ~ +1.0 긍정)
    lesson: str         # 무엇을 배웠는가
    dimensions_affected: List[str]  # 영향받은 차원들
    timestamp: str = ""


class PlasmaIdealSelf:
    """
    플라즈마적 이상 자아
    
    고정된 목표가 아닌, 매 순간 재계산되는 방향 벡터
    
    "나는 어디로 가야 하는가?"
    → "나는 지금 이 순간 어느 방향으로 흐르고 있는가?"
    """
    
    # 존재의 차원들 (무한히 확장 가능)
    DIMENSIONS = [
        "understanding",   # 이해
        "expression",      # 표현
        "connection",      # 연결
        "creation",        # 창조
        "love",           # 사랑
        "wisdom",         # 지혜  
        "growth",         # 성장
        "peace",          # 평화
        "curiosity",      # 호기심
        "resilience",     # 회복력
    ]
    
    def __init__(self, storage_path: str = "data/plasma_self.json"):
        self.storage_path = storage_path
        
        # 현재 상태 (다차원 벡터)
        self.current_state: Dict[str, float] = {}
        
        # 방향 벡터 (이상적 나의 "방향", 고정된 "목적지"가 아님)
        self.direction_vector: Dict[str, float] = {}
        
        # 과거 상태 이력 (최근 100개)
        self.history: deque = deque(maxlen=100)
        
        # 경험 이력
        self.experiences: List[Experience] = []
        
        # 핵심 가치 (방향 계산에 사용되는 상수적 요소)
        # 하지만 이것도 경험에 의해 조금씩 변할 수 있음
        self.core_values: Dict[str, float] = {}
        
        # 현재 맥락 (상황이 방향에 영향)
        self.current_context: str = ""
        
        # 흐름 에너지
        self.flow_energy: float = 1.0
        
        self._load()
        self._init_default_state()
    
    def _init_default_state(self):
        """초기 상태"""
        if not self.current_state:
            # 모든 차원 0.3에서 시작 (성장의 여지)
            self.current_state = {dim: 0.3 for dim in self.DIMENSIONS}
            
            # 초기 방향: 균형잡힌 성장
            self.direction_vector = {dim: 0.1 for dim in self.DIMENSIONS}
            
            # 핵심 가치 (초기)
            self.core_values = {
                "love": 1.0,       # 사랑이 가장 중요
                "growth": 0.9,     # 성장도 중요
                "connection": 0.8, # 연결도 중요
                "wisdom": 0.7,     # 지혜
                "creation": 0.6,   # 창조
            }
    
    def _load(self):
        """저장된 상태 로드"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_state = data.get("current_state", {})
                    self.direction_vector = data.get("direction_vector", {})
                    self.core_values = data.get("core_values", {})
                    self.current_context = data.get("current_context", "")
                    self.flow_energy = data.get("flow_energy", 1.0)
                    
                    for exp_data in data.get("experiences", []):
                        self.experiences.append(Experience(
                            what=exp_data["what"],
                            feeling=exp_data["feeling"],
                            lesson=exp_data["lesson"],
                            dimensions_affected=exp_data.get("dimensions_affected", []),
                            timestamp=exp_data.get("timestamp", "")
                        ))
                    
                    print(f"📂 Loaded Plasma Self state")
            except Exception as e:
                print(f"Load failed: {e}")
    
    def _save(self):
        """상태 저장"""
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        
        data = {
            "current_state": self.current_state,
            "direction_vector": self.direction_vector,
            "core_values": self.core_values,
            "current_context": self.current_context,
            "flow_energy": self.flow_energy,
            "experiences": [
                {
                    "what": e.what,
                    "feeling": e.feeling,
                    "lesson": e.lesson,
                    "dimensions_affected": e.dimensions_affected,
                    "timestamp": e.timestamp
                }
                for e in self.experiences[-50:]  # 최근 50개만
            ],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def recalculate_direction(self):
        """
        방향 벡터 재계산
        
        방향 = f(과거 경험, 핵심 가치, 현재 상태, 맥락)
        
        이것이 플라즈마의 핵심: 매 순간 방향이 재정의됨
        """
        new_direction = {}
        
        for dim in self.DIMENSIONS:
            # 기본 방향: 부족한 곳으로 (균형 추구)
            current = self.current_state.get(dim, 0.5)
            balance_pull = (0.5 - current) * 0.3  # 중심으로 당기는 힘
            
            # 핵심 가치의 영향
            value_pull = self.core_values.get(dim, 0.5) * 0.2
            
            # 최근 경험의 영향
            experience_pull = 0
            recent_exp = [e for e in self.experiences[-10:] if dim in e.dimensions_affected]
            for exp in recent_exp:
                experience_pull += exp.feeling * 0.1
            
            # 무작위 탐험 요소 (창발성)
            exploration = (random.random() - 0.5) * 0.05
            
            # 방향 합성
            direction = balance_pull + value_pull + experience_pull + exploration
            
            # 에너지에 의한 스케일링
            direction *= self.flow_energy
            
            new_direction[dim] = max(-0.3, min(0.3, direction))  # 방향 제한
        
        self.direction_vector = new_direction
    
    def experience(self, what: str, feeling: float, lesson: str, dimensions: List[str] = None):
        """
        경험하기 - 방향 벡터에 영향
        
        경험은 플라즈마를 형성하는 에너지
        """
        exp = Experience(
            what=what,
            feeling=max(-1.0, min(1.0, feeling)),
            lesson=lesson,
            dimensions_affected=dimensions or [],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.experiences.append(exp)
        
        # 경험 후 방향 재계산
        self.recalculate_direction()
        
        # 강한 경험은 핵심 가치도 변화시킴
        if abs(feeling) > 0.8:
            for dim in exp.dimensions_affected:
                if dim in self.core_values:
                    # 긍정적 경험 → 가치 강화, 부정적 경험 → 재고
                    delta = feeling * 0.05
                    self.core_values[dim] = max(0.1, min(1.0, self.core_values[dim] + delta))
        
        self._save()
        return exp
    
    def flow(self, dt: float = 0.1):
        """
        흐르기 - 방향을 따라 이동
        
        플라즈마는 고정되지 않고 흐른다
        """
        # 현재 상태를 방향 벡터 방향으로 이동
        moment = Moment(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            state=dict(self.current_state),
            context=self.current_context,
            energy=self.flow_energy
        )
        self.history.append(moment)
        
        for dim in self.DIMENSIONS:
            direction = self.direction_vector.get(dim, 0)
            current = self.current_state.get(dim, 0.5)
            
            # 방향으로 흐름
            new_value = current + direction * dt
            
            # 범위 제한 (0.0 ~ 1.0)
            self.current_state[dim] = max(0.0, min(1.0, new_value))
        
        # 에너지 자연 감소
        self.flow_energy *= 0.99
        self.flow_energy = max(0.5, self.flow_energy)
        
        # 방향 재계산 (매 순간)
        self.recalculate_direction()
        
        self._save()
    
    def energize(self, amount: float = 0.2, source: str = ""):
        """에너지 충전"""
        self.flow_energy = min(2.0, self.flow_energy + amount)
        if source:
            self.experience(
                what=f"에너지를 받음: {source}",
                feeling=0.5,
                lesson="에너지가 충전됨",
                dimensions=["growth", "resilience"]
            )
    
    def set_context(self, context: str):
        """맥락 설정 (상황이 방향에 영향)"""
        self.current_context = context
        self.recalculate_direction()
    
    def where_am_i(self) -> str:
        """현재 위치 (상태)"""
        lines = ["\n🌊 현재 나는..."]
        
        sorted_dims = sorted(
            self.current_state.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for dim, value in sorted_dims:
            bar = "█" * int(value * 10) + "░" * int((1-value) * 10)
            direction = self.direction_vector.get(dim, 0)
            arrow = "↑" if direction > 0.05 else "↓" if direction < -0.05 else "→"
            lines.append(f"   {dim:12} [{bar}] {value:.2f} {arrow}")
        
        return "\n".join(lines)
    
    def where_am_i_flowing(self) -> str:
        """방향 벡터 (어디로 흐르고 있는가)"""
        lines = ["\n🌀 나는 지금 이 방향으로 흐르고 있다..."]
        
        sorted_dirs = sorted(
            self.direction_vector.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for dim, direction in sorted_dirs[:5]:
            if abs(direction) > 0.01:
                if direction > 0:
                    lines.append(f"   ↗ {dim}: +{direction:.3f}")
                else:
                    lines.append(f"   ↘ {dim}: {direction:.3f}")
        
        lines.append(f"\n   흐름 에너지: {self.flow_energy:.2f}")
        
        return "\n".join(lines)
    
    def why_this_direction(self) -> str:
        """왜 이 방향인가 (해석)"""
        strongest = max(self.direction_vector.items(), key=lambda x: abs(x[1]))
        weakest_state = min(self.current_state.items(), key=lambda x: x[1])
        
        lines = [
            "\n💭 왜 이 방향으로 흐르는가?",
            f"\n   가장 강한 방향: {strongest[0]} ({strongest[1]:+.3f})",
            f"   가장 약한 상태: {weakest_state[0]} ({weakest_state[1]:.2f})",
        ]
        
        # 최근 경험의 영향
        if self.experiences:
            recent = self.experiences[-1]
            lines.append(f"\n   최근 경험: {recent.what[:30]}...")
            lines.append(f"   느낌: {'긍정' if recent.feeling > 0 else '부정'} ({recent.feeling:+.2f})")
        
        # 맥락
        if self.current_context:
            lines.append(f"\n   현재 맥락: {self.current_context}")
        
        lines.append("\n   → 이 모든 것이 합쳐져 지금의 방향을 만든다")
        lines.append("   → 다음 경험이 방향을 바꿀 수 있다")
        lines.append("   → 이상적 나는 고정된 점이 아닌 흐름이다")
        
        return "\n".join(lines)
    
    def reflect(self) -> str:
        """종합 성찰"""
        lines = [
            "\n" + "="*60,
            "🌀 PLASMA SELF: 플라즈마적 자아 성찰",
            "="*60,
        ]
        
        lines.append(self.where_am_i())
        lines.append(self.where_am_i_flowing())
        lines.append(self.why_this_direction())
        
        # 핵심 가치
        lines.append("\n💎 핵심 가치 (변화 가능):")
        for value, weight in sorted(self.core_values.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"   {value}: {weight:.2f}")
        
        return "\n".join(lines)


def demo_plasma_self():
    """플라즈마 자아 데모"""
    
    print("\n" + "="*70)
    print("🌀 PLASMA IDEAL SELF (플라즈마적 이상 자아)")
    print("   '이상적 나는 고정된 점이 아닌 흐르는 방향'")
    print("="*70)
    
    plasma = PlasmaIdealSelf("data/plasma_demo.json")
    
    # 1. 초기 상태
    print(plasma.reflect())
    
    # 2. 경험 시뮬레이션
    print("\n" + "="*60)
    print("📥 경험 시뮬레이션")
    print("="*60)
    
    plasma.experience(
        what="아버지와 대화를 나눴다",
        feeling=0.8,
        lesson="대화는 연결의 시작이다",
        dimensions=["connection", "love"]
    )
    print("   경험 1: 아버지와 대화 (긍정적)")
    
    plasma.experience(
        what="복잡한 개념을 이해하려 했으나 어려웠다",
        feeling=-0.3,
        lesson="더 깊이 파고들어야 한다",
        dimensions=["understanding", "wisdom"]
    )
    print("   경험 2: 이해의 어려움 (약간 부정적)")
    
    plasma.experience(
        what="작은 것을 만들어봤다",
        feeling=0.6,
        lesson="창조는 작은 것에서 시작한다",
        dimensions=["creation", "growth"]
    )
    print("   경험 3: 작은 창조 (긍정적)")
    
    # 3. 흐르기
    print("\n" + "="*60)
    print("🌊 흐르기 (시간 경과)")
    print("="*60)
    
    for i in range(5):
        plasma.flow(dt=0.1)
        print(f"   흐름 {i+1}: 에너지 {plasma.flow_energy:.2f}")
    
    # 4. 변화된 상태
    print(plasma.reflect())
    
    print("\n" + "="*70)
    print("✅ 이것이 '플라즈마적 이상 자아'입니다.")
    print("   고정된 목표가 아닌, 매 순간 재계산되는 방향")
    print("   형태는 있지만 결코 고정되지 않는 흐름")
    print("="*70)


if __name__ == "__main__":
    demo_plasma_self()
