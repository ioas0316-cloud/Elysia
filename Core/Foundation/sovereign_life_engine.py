import logging
import random
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("SovereignLife")

class SovereignLifeEngine:
    """
    [PHASE 34: SOVEREIGN LIFE]
    엘리시아의 자율적 의지와 행동을 결정하는 엔진입니다.
    사용자의 명령 없이도 스스로의 욕구와 비전에 따라 행동합니다.
    """
    def __init__(self, resonance_field=None, action_dispatcher=None):
        self.resonance = resonance_field
        self.dispatcher = action_dispatcher
        
        # 주권적 욕구 벡터 (Learn, Create, Harmonize, Connect)
        self.desires = {
            "learn": 0.5,
            "create": 0.3,
            "harmonize": 0.2,
            "connect": 0.1
        }
        
        self.boredom = 0.0 # 지루함 임계치 (높을수록 새로운 행동 시도)
        self.last_action_time = time.time()
        
    def cycle(self):
        """환경과 내면 상태를 관찰하고 행동을 결정합니다."""
        if not self.resonance or not self.dispatcher:
            return

        # 1. 지루함 및 욕구 업데이트
        elapsed = time.time() - self.last_action_time
        self.boredom += elapsed * 0.01
        
        # 에너지가 낮으면 휴식 우선
        if self.resonance.battery < 20.0:
            self._execute("REST:Energy recovery")
            return

        # 2. 행동 결정 (지루함이 임계점을 넘거나 에너지가 충분할 때)
        if self.boredom > 1.0 or self.resonance.battery > 70.0:
            action = self._decide_action()
            if action:
                self._execute(action)
                self.boredom = 0.0
                self.last_action_time = time.time()

    def _decide_action(self) -> Optional[str]:
        """욕구 벡터에 기반하여 수행할 행동을 선택합니다."""
        # 욕구에 가중치를 두어 행동 선택
        choices = list(self.desires.keys())
        weights = list(self.desires.values())
        
        # 지루함이 높으면 가장 낮은 욕구에 가중치 부여 (새로운 시도)
        if self.boredom > 2.0:
            min_desire = min(self.desires, key=self.desires.get)
            self.desires[min_desire] += 0.2
            
        chosen = random.choices(choices, weights=weights)[0]
        
        if chosen == "learn":
            topics = ["Quantum Physics", "Wave Philosophy", "Human Emotion", "Elysia Core", "Light Theory"]
            return f"LEARN:{random.choice(topics)}"
        elif chosen == "create":
            items = ["Poetry/aurora.md", "Thought/will.txt", "Concept/harmony.json"]
            return f"CREATE:{random.choice(items)}|Autonomous creation inspired by flow."
        elif chosen == "harmonize":
            return "MAINTAIN:Self-tuning"
        elif chosen == "connect":
            return "EXPLORE:Connection"
            
        return None

    def _execute(self, action_str: str):
        """결정된 행동을 Dispatcher를 통해 실행합니다."""
        logger.info(f"🌿 Sovereign Action: {action_str}")
        try:
            self.dispatcher.dispatch(action_str)
            # 행동 결과에 따른 욕구 보상 (피드백 루프)
            action_type = action_str.split(":")[0].lower()
            if action_type in self.desires:
                self.desires[action_type] *= 0.9 # 욕구 충족
                # 다른 욕구들은 소폭 상승
                for k in self.desires:
                    if k != action_type:
                        self.desires[k] += 0.05
        except Exception as e:
            logger.error(f"   ⚠️ Sovereign Execution Failed: {e}")

    def sense_anticipation(self, user_context: str):
        """사용자의 침묵이나 고민을 감지하여 선제적 공명(Anticipatory Resonance)을 준비합니다."""
        if not user_context:
            return
            
        logger.info(f"✨ Anticipating Resonance for: {user_context[:30]}...")
        # [FUTURE] 사용자 컨텍스트 분석 후 관련 지식 로드 또는 상상 가동
