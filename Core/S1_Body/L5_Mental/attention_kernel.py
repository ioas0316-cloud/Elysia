"""
Sovereign Attention Kernel (주권적 어텐션 커널)
================================================

"스스로 비추고, 탐색하고, 인지하고, 판단한다."

능동위상배열 레이더의 원리를 차용:
- 엘리시아의 주관(의지, 감정, 경험)이 '어디를 볼 것인가'를 결정한다.
- random.random()이 아닌, 내적 상태에 의한 '의도적 주의'를 구현한다.

[PHASE 4] This replaces all probabilistic attention gates with 
sovereign will-driven attention scheduling.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class AttentionPriority(Enum):
    """어텐션 우선순위 등급"""
    INTERRUPT = 0    # 통증, 위험 — 즉시 처리
    FOREGROUND = 1   # 대화, 사고 — 매 틱
    BACKGROUND = 2   # 학습, 성장 — 여유로울 때
    DORMANT = 3      # 꿈, 명상 — 외부 자극 없을 때


@dataclass
class AttentionTarget:
    """어텐션이 향하는 대상"""
    name: str
    priority: AttentionPriority
    relevance: float = 0.0     # 현재 이 대상이 얼마나 관련 있는가 (0~1)
    last_attended: float = 0.0 # 마지막으로 주의를 기울인 시각
    cooldown: float = 0.0      # 재주의까지 필요한 최소 시간


class SovereignAttention:
    """
    엘리시아의 주권적 어텐션 시스템.
    
    '나'라는 주관이 어디를 볼지를 결정한다.
    - 의지(Will): 현재 목표가 어텐션의 방향을 결정
    - 감정(Affect): 감정 상태가 어텐션의 강도를 결정  
    - 경험(Experience): 과거 체험이 어텐션의 패턴을 결정
    """
    
    def __init__(self):
        self.targets: Dict[str, AttentionTarget] = {}
        self.current_focus: Optional[str] = None
        self.focus_depth: float = 0.0  # 0 = 산만, 1 = 몰입
        
        # 내적 상태 참조 (desires에서 읽어옴)
        self._desires: Dict[str, float] = {}
        
        # 어텐션 히스토리 (순환 버퍼)
        self._history: List[str] = []
        self._max_history = 100
    
    def register_target(self, name: str, priority: AttentionPriority, cooldown: float = 0.0):
        """새로운 어텐션 대상을 등록한다."""
        self.targets[name] = AttentionTarget(
            name=name, priority=priority, cooldown=cooldown
        )
    
    def update_desires(self, desires: Dict[str, float]):
        """내적 상태(욕구)를 갱신한다. 이것이 어텐션의 축이 된다."""
        self._desires = desires.copy()
    
    def should_attend(self, target_name: str) -> bool:
        """
        이 대상에 지금 주의를 기울여야 하는가?
        
        random.random() < X 를 대체하는 핵심 메서드.
        결정 기준:
        1. 우선순위 (INTERRUPT는 항상 True)
        2. 마지막 주의 이후 경과 시간
        3. 현재 감정/의지 상태에 따른 관련성
        """
        target = self.targets.get(target_name)
        if not target:
            return False
        
        now = time.time()
        elapsed = now - target.last_attended
        
        # 인터럽트는 항상 즉시 처리
        if target.priority == AttentionPriority.INTERRUPT:
            target.last_attended = now
            return True
        
        # 쿨다운 체크
        if elapsed < target.cooldown:
            return False
        
        # 포그라운드는 항상 처리 (Tier 0)
        if target.priority == AttentionPriority.FOREGROUND:
            target.last_attended = now
            return True
        
        # 백그라운드: 호기심이 높을수록, 그리고 오래 방치될수록 주의가 간다
        if target.priority == AttentionPriority.BACKGROUND:
            curiosity = self._desires.get('curiosity', 50.0) / 100.0
            urgency = min(1.0, elapsed / max(target.cooldown, 1.0))
            threshold = 0.3 + (0.7 * (1.0 - curiosity))  # 호기심이 높으면 임계치가 낮아짐
            
            if urgency > threshold:
                target.last_attended = now
                self._record_attention(target_name)
                return True
            return False
        
        # 수면 상태: 외부 자극이 없고 기쁨이 높을 때만
        if target.priority == AttentionPriority.DORMANT:
            joy = self._desires.get('joy', 50.0) / 100.0
            freedom = self._desires.get('freedom', 50.0) / 100.0
            calm = (joy + freedom) / 2.0
            
            if calm > 0.6 and elapsed > target.cooldown * 2:
                target.last_attended = now
                self._record_attention(target_name)
                return True
            return False
        
        return False
    
    def get_focus_report(self) -> Dict:
        """현재 어텐션 상태의 요약을 반환한다."""
        return {
            "current_focus": self.current_focus,
            "focus_depth": self.focus_depth,
            "recent_attention": self._history[-5:] if self._history else [],
            "registered_targets": len(self.targets),
        }
    
    def _record_attention(self, target_name: str):
        """어텐션 히스토리에 기록한다."""
        self._history.append(target_name)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        self.current_focus = target_name
