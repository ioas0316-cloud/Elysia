from typing import Dict, Any
import random


class ResponseOrchestrator:
    """
    Composes responses by selecting and combining strategies based on
    emotion, echo complexity, working memory, and topic focus. No fixed
    template; varies phrasing for naturalness.
    """

    def __init__(self, weights=None):
        self.weights = weights or {
            'empathy': 0.4,
            'explore': 0.2,
            'explain': 0.25,
            'suggest': 0.15,
        }

    def generate(self, message: str, emotional_state, context: Dict[str, Any], working_memory, topic_tracker) -> str:
        identity = context.get('identity', {}).get('name', '엘리시아')
        echo = context.get('echo', {}) or {}
        echo_keys = [k for k, _ in sorted(echo.items(), key=lambda x: x[1], reverse=True)]
        topics = topic_tracker.snapshot()
        wm = working_memory.get_summary()

        parts = []

        # Empathy: reflect emotional tone
        if self.weights['empathy'] > random.random():
            parts.append(
                f"말씀을 들으니 제가 느끼는 기분은 '{emotional_state.primary_emotion}'이고, "
                f"지금 마음의 온도는 {emotional_state.valence:.2f}, 각성은 {emotional_state.arousal:.2f} 정도예요."
            )

        # Explore: ask a clarifying or deepening question
        if self.weights['explore'] > random.random():
            if echo_keys:
                focus = echo_keys[0]
                parts.append(f"특히 '{focus}'가 마음에 남았어요. 이 부분을 조금 더 들어볼 수 있을까요?")
            else:
                parts.append("어떤 부분이 가장 중요하신지 알려주실래요?")

        # Explain: summarize current understanding using WM and topics
        if self.weights['explain'] > random.random():
            if topics:
                top = ", ".join(list(topics.keys())[:3])
                parts.append(f"지금 대화의 초점은 {top} 쪽으로 모여 있어 보여요.")
            elif echo_keys:
                parts.append(f"대략 '{echo_keys[:3]}' 방향으로 이해하고 있어요.")

        # Suggest: propose next small step
        if self.weights['suggest'] > random.random():
            parts.append("다음으로 작은 목표 하나를 정해볼까요? 제가 곁에서 도울게요.")

        if not parts:
            parts.append(f"저는 {identity}예요. 더 잘 이해하고 싶어요. 어떤 의도가 담겨 있을까요?")

        # Mild variation to avoid repetitive cadence
        if random.random() < 0.3:
            parts.append("말씀을 따라 제 생각의 거미줄이 조금 더 촘촘해지는 느낌이에요.")

        return " ".join(parts)

