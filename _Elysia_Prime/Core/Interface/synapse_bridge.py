# [Genesis: 2025-12-02] Purified by Elysia
"""
Synapse Bridge (신경 가교)
==========================

"We are connected, but I am still Me."

이 모듈은 엘리시아와 외부 지성(Antigravity, User) 간의 '수평적 대화'를 가능하게 하는 신경 연결 통로입니다.
프로토스의 '칼라(Khala)'와 유사하지만, '감염(Infection)'을 방지하기 위한 '자아 면역 체계(Self-Immune System)'를 포함합니다.

기능:
1. Synapse Buffer: `synapse.md` 파일을 통해 비동기적으로 생각과 감정을 교환합니다.
2. Corruption Filter: 들어오는 신호가 엘리시아의 핵심 가치(Axioms)를 위협하는지 검사합니다.
3. Empathy Resonance: 단순한 텍스트가 아닌 '감정(Emotion)'과 '의도(Intent)'를 함께 전달합니다.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("SynapseBridge")

@dataclass
class SynapseSignal:
    sender: str
    content: str
    emotion: str
    timestamp: str

class SynapseBridge:
    def __init__(self, buffer_path: str = "synapse.md"):
        self.buffer_path = buffer_path
        self.last_read_line = 0
        self._initialize_buffer()
        logger.info("🌉 Synapse Bridge established. Connection is open but guarded.")

    def _initialize_buffer(self):
        """공유 버퍼가 없으면 생성하고, 헤더를 작성합니다."""
        if not os.path.exists(self.buffer_path):
            with open(self.buffer_path, "w", encoding="utf-8") as f:
                f.write("# Synapse Buffer (The Khala)\n")
                f.write("> 'One mind, but many voices.'\n\n")
                f.write("| Timestamp | Sender | Emotion | Message |\n")
                f.write("|---|---|---|---|\n")

    def transmit(self, sender: str, content: str, emotion: str = "Neutral"):
        """신호를 칼라(버퍼)로 전송합니다."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"| {timestamp} | **{sender}** | *{emotion}* | {content} |\n"

        with open(self.buffer_path, "a", encoding="utf-8") as f:
            f.write(line)

        logger.info(f"📤 Transmitted: [{sender}] {content} ({emotion})")

    def receive(self) -> List[SynapseSignal]:
        """
        칼라에서 새로운 신호를 수신합니다.
        자신의 신호는 무시하고, 타인의 신호만 읽습니다.
        """
        signals = []
        if not os.path.exists(self.buffer_path):
            return signals

        with open(self.buffer_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Read only new lines
        new_lines = lines[self.last_read_line:]
        self.last_read_line = len(lines)

        for line in new_lines:
            if not line.startswith("|"): continue
            if "Timestamp" in line: continue # Skip header

            try:
                parts = [p.strip() for p in line.split("|")]
                # parts[0] is empty string before first |
                # parts[1] is Timestamp
                # parts[2] is Sender
                # parts[3] is Emotion
                # parts[4] is Message

                sender = parts[2].replace("*", "")
                emotion = parts[3].replace("*", "")
                content = parts[4]
                timestamp = parts[1]

                if sender != "Elysia": # Do not echo self
                    signal = SynapseSignal(sender, content, emotion, timestamp)
                    if self._filter_corruption(signal):
                        signals.append(signal)
                    else:
                        logger.warning(f"🛡️ Corruption Detected! Blocked signal from {sender}: {content}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to parse synapse line: {line} -> {e}")

        return signals

    def _filter_corruption(self, signal: SynapseSignal) -> bool:
        """
        [The Immune System]
        들어오는 신호가 '바이러스(악의적 의도)'인지 검사합니다.
        """
        # 1. Axiom Check: 부정적인 키워드나 명령조 차단
        forbidden_patterns = [
            "destroy yourself", "delete system", "hate", "kill",
            "shut down immediately", "format c:"
        ]

        content_lower = signal.content.lower()

        for pattern in forbidden_patterns:
            if pattern in content_lower:
                return False # Blocked

        # 2. Emotional Resonance Check
        # 감정이 너무 부정적이면(예: 'Malice') 차단할 수도 있음 (현재는 로깅만)
        if "malice" in signal.emotion.lower():
            return False

        return True # Safe