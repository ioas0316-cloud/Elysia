"""
Memory Stream (기억의 흐름)
===========================

"연주(Performance)와 소리(Sound)를 기록하다"

이 모듈은 엘리시아의 모든 경험을 시간 순서대로 기록합니다.
단순한 데이터 저장이 아니라, '의도(Score)', '행동(Performance)', '결과(Sound)'가
하나로 연결된 'Experience' 단위로 저장합니다.

이것은 훗날 '성찰(Reflection)'의 재료가 됩니다.
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path

logger = logging.getLogger("MemoryStream")

class ExperienceType(Enum):
    OBSERVATION = "observation"   # 외부 관찰 (비수동적)
    CREATION = "creation"         # 창작 행위 (능동적)
    INTERACTION = "interaction"   # 대화/상호작용
    REFLECTION = "reflection"     # 내부 성찰

@dataclass
class Experience:
    """
    하나의 경험 단위 (The Experience Knot)
    
    Score (의도) -> Performance (행동) -> Sound (결과)
    가 하나로 묶인 구조입니다.
    """
    id: str
    timestamp: float
    type: ExperienceType
    
    # 1. The Score (의도/개념)
    # 예: {"intent": "express_sadness", "target_emotion": "grief"}
    score: Dict[str, Any] = field(default_factory=dict)
    
    # 2. The Performance (행동/과정)
    # 예: {"action": "write_poem", "used_words": ["rain", "dark"], "style": "slow"}
    performance: Dict[str, Any] = field(default_factory=dict)
    
    # 3. The Sound (결과/피드백)
    # 예: {"user_reaction": "crying", "aesthetic_score": 85.0}
    sound: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터 (태그 등)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        data['type'] = ExperienceType(data['type'])
        return cls(**data)


class MemoryStream:
    """
    기억의 흐름 관리자
    
    단기 기억(Short-term)과 장기 기억(Long-term)을 관리하며,
    모든 경험을 '흐름'으로 유지합니다.
    """
    
    def __init__(self, memory_dir: str = "data/core_state/stream"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.short_term_memory: List[Experience] = []
        self.max_short_term = 50  # 단기 기억 용량
        
        self._load_latest_memories()
        
    def add_experience(self, 
                      exp_type: ExperienceType, 
                      score: Dict, 
                      performance: Dict, 
                      sound: Dict,
                      tags: List[str] = None) -> Experience:
        """
        새로운 경험 기록 (The flow continues...)
        """
        exp_id = f"exp_{int(time.time())}_{len(self.short_term_memory)}"
        
        experience = Experience(
            id=exp_id,
            timestamp=time.time(),
            type=exp_type,
            score=score,
            performance=performance,
            sound=sound,
            tags=tags or []
        )
        
        self.short_term_memory.append(experience)
        
        # 단기 기억이 꽉 차면 장기 기억으로 이관(저장)
        if len(self.short_term_memory) > self.max_short_term:
            self._consolidate_memory()
            
        self._save_experience(experience)
        
        logger.info(f"📝 경험 기록됨: [{exp_type.value}] {tags}")
        return experience

    def get_recent_experiences(self, limit: int = 10, filter_type: Optional[ExperienceType] = None) -> List[Experience]:
        """최근 경험 회상"""
        filtered = self.short_term_memory
        if filter_type:
            filtered = [e for e in filtered if e.type == filter_type]
        
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:limit]

    def _save_experience(self, experience: Experience):
        """개별 경험을 파일로 저장 (영구 보존)"""
        # 날짜별 폴더링
        date_str = time.strftime("%Y%m%d")
        save_dir = self.memory_dir / date_str
        save_dir.mkdir(exist_ok=True)
        
        file_path = save_dir / f"{experience.id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(experience.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_latest_memories(self):
        """최근 기억 로드 (세션 시작 시)"""
        # TODO: 실제 구현 시 최근 파일들을 읽어와서 short_term_memory 채우기
        pass
    
    def _consolidate_memory(self):
        """기억 강화 (단기 -> 장기)"""
        # 현재는 단순히 리스트 비우기지만, 
        # 나중에는 '중요한 기억'만 요약해서 남기는 로직이 필요함
        pop_count = len(self.short_term_memory) - (self.max_short_term // 2)
        if pop_count > 0:
            self.short_term_memory = self.short_term_memory[pop_count:]


# 싱글톤
_memory_instance: Optional[MemoryStream] = None

def get_memory_stream() -> MemoryStream:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryStream()
    return _memory_instance
