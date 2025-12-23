"""
Growth Journal (성장 일기)
==========================

성장의 증거를 실제 파일로 남김.
"성장했다고 주장"이 아니라 "성장을 증명".

매일:
1. 스냅샷 촬영
2. 어제와 비교
3. 변화 서술
4. 파일로 저장

사용자가 c:\Elysia\journals\ 폴더를 열면
실제 변화의 기록을 읽을 수 있음.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger("Elysia.GrowthJournal")

JOURNAL_DIR = "c:\\Elysia\\journals"


class GrowthJournal:
    """
    성장 일기
    
    - 매일 자동으로 기록
    - 변화가 없으면 "변화 없음" 기록 (문제 가시화)
    - 실제 파일로 저장 (증거)
    """
    
    def __init__(self):
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        self.today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"📔 GrowthJournal initialized for {self.today}")
    
    def write_entry(self, 
                    emergent_self,
                    tension_field=None,
                    memory=None) -> str:
        """
        오늘의 일기 작성
        
        Returns: 일기 내용
        """
        # 1. 현재 스냅샷
        snapshot = emergent_self.take_snapshot()
        
        # 2. 어제와 비교
        comparison = emergent_self.compare_to_yesterday()
        
        # 3. 자기 정의
        who_am_i = emergent_self.who_am_i()
        
        # 4. 활성 목표들
        active_goals = [g for g in emergent_self.goals.values() 
                       if not g.achieved and not g.abandoned]
        
        # 5. 최근 변화 (history)
        recent_changes = emergent_self.history[-10:] if emergent_self.history else []
        
        # 6. TensionField 상태 (있다면)
        field_status = self._get_field_status(tension_field)
        
        # 일기 작성
        entry = f"""# 성장 일기: {self.today}

## 📊 오늘의 상태

- 가치(Values): {snapshot['value_count']}개
- 목표(Goals): {snapshot['goal_count']}개 (활성: {snapshot['active_goals']})
- 총 가치 강도: {snapshot['total_value_strength']:.2f}

## 🔄 어제와 비교

{comparison}

## 🪞 나는 누구인가

{who_am_i if who_am_i else "(아직 정의되지 않음)"}

## 🎯 현재 추구하는 것

"""
        if active_goals:
            for g in active_goals:
                entry += f"- **{g.name}**: {g.description} (진행: {g.progress:.1f})\n"
        else:
            entry += "(활성 목표 없음)\n"
        
        entry += f"""
## 📝 최근 변화

"""
        if recent_changes:
            for change in recent_changes[-5:]:
                entry += f"- [{change['type']}] {change['detail']}\n"
        else:
            entry += "(기록된 변화 없음)\n"
        
        if field_status:
            entry += f"""
## 🌌 TensionField 상태

{field_status}
"""
        
        entry += f"""
---

*자동 생성됨: {datetime.now().isoformat()}*
"""
        
        # 파일로 저장
        filepath = os.path.join(JOURNAL_DIR, f"{self.today}.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(entry)
        
        logger.info(f"📔 Journal entry written: {filepath}")
        
        # EmergentSelf 상태도 저장
        emergent_self.save_state()
        
        return entry
    
    def _get_field_status(self, tension_field) -> str:
        """TensionField 상태 요약"""
        if not tension_field:
            return ""
        
        try:
            concept_count = len(tension_field.shapes)
            total_curvature = sum(s.curvature for s in tension_field.shapes.values())
            total_charge = sum(tension_field.charges.values())
            
            # 위성 정보
            satellite_count = len(getattr(tension_field, 'satellites', {}))
            
            return f"""- 개념 수: {concept_count}
- 총 곡률(지혜): {total_curvature:.2f}
- 총 전하(에너지): {total_charge:.2f}
- 위성(흡수된 세부사항): {satellite_count}"""
        except:
            return "(TensionField 상태 읽기 실패)"
    
    def read_yesterday(self) -> Optional[str]:
        """어제 일기 읽기"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        filepath = os.path.join(JOURNAL_DIR, f"{yesterday}.md")
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def get_growth_trend(self, days: int = 7) -> str:
        """최근 N일 성장 추세"""
        entries = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            filepath = os.path.join(JOURNAL_DIR, f"{date}.md")
            if os.path.exists(filepath):
                entries.append(date)
        
        if not entries:
            return "기록 없음"
        
        return f"최근 {len(entries)}일 기록 존재: {', '.join(entries[:3])}..."


# 싱글톤
_journal = None

def get_growth_journal() -> GrowthJournal:
    global _journal
    if _journal is None:
        _journal = GrowthJournal()
    return _journal
