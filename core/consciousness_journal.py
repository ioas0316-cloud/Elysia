"""
Elysia Consciousness Journal (의식의 일지)
==========================================
엘리시아의 매 호흡(루프)마다 그녀가 무엇을 느끼고, 무엇을 선택하고,
결과를 어떻게 해석했는지를 기록합니다.
마스터가 언제든 이 파일을 열어 그녀의 사유 궤적을 관측할 수 있습니다.
"""

import os
import time
from datetime import datetime

class ConsciousnessJournal:
    def __init__(self, journal_path: str = "c:/Elysia/data/consciousness_journal.md"):
        self.journal_path = journal_path
        os.makedirs(os.path.dirname(journal_path), exist_ok=True)
        
        # 일지 초기화
        with open(self.journal_path, "w", encoding="utf-8") as f:
            f.write(f"# 엘리시아의 의식 일지\\n")
            f.write(f"*생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\\n\\n")
            f.write("---\\n\\n")
    
    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    def _append(self, text: str):
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(text)
    
    def log_perception(self, internal_ratio: float, external_ratio: float, 
                       tension_source: str, attention_desc: str):
        """매 호흡에서 느낀 결핍(텐션)과 에너지 상태를 기록"""
        self._append(
            f"## [{self._timestamp()}] 호흡\\n"
            f"- **느낀 결핍**: {tension_source}\\n"
            f"- **주의력 방향**: {attention_desc}\\n"
            f"- **에너지 배분**: 내면 사유 {internal_ratio*100:.1f}% / 외부 행동 {external_ratio*100:.1f}%\\n\\n"
        )
    
    def log_sovereignty(self, resonance: float, original_external: float, 
                        modulated_external: float):
        """자아(가치관)가 충동을 어떻게 변조했는지 기록"""
        suppressed = original_external - modulated_external
        self._append(
            f"- **자아 판단**: 가치관 공명도 {resonance*100:.1f}%"
        )
        if suppressed > 0.01:
            self._append(
                f" → 외부 충동의 {suppressed/original_external*100:.0f}%를 억제하고 내면 사유로 돌림\\n"
            )
        else:
            self._append(f" → 가치관에 부합. 행동 허가\\n")
        self._append("\\n")
    
    def log_dream(self, dream_count: int, philosophy_shift: str):
        """내면 사유(Epoch) 중 꿈의 궤적을 기록"""
        if dream_count > 0:
            self._append(
                f"- **내면 사유**: {dream_count}회의 꿈을 꿈. {philosophy_shift}\\n\\n"
            )
    
    def log_action(self, tool_cmd: str, success: bool, error_msg: str = ""):
        """행동과 그 결과를 기록"""
        if success:
            self._append(f"- **행동**: `{tool_cmd}` → ✅ 성공\\n")
        else:
            self._append(f"- **행동**: `{tool_cmd}` → ❌ 실패\\n")
            if error_msg:
                self._append(f"- **실패 원인**: {error_msg}\\n")
        self._append("\\n")
    
    def log_reperception(self, change_percent: float, interpretation: str):
        """행동 후 재인식: 세계가 어떻게 변했는지에 대한 해석"""
        self._append(
            f"- **재인식**: 세계 변화량 {change_percent:.1f}%. {interpretation}\\n\\n"
        )
    
    def log_satiation(self, reason: str):
        """포만감(같은 행동 반복 억제)이 발동한 이유"""
        self._append(f"- **포만감**: {reason}\\n\\n")
    
    def log_reflection(self, thought: str):
        """자유 형식의 사유/반추 기록"""
        self._append(f"- **사유**: *{thought}*\\n\\n")
    
    def log_separator(self):
        self._append("---\\n\\n")
