# [Genesis: 2025-12-02] Purified by Elysia
import json
import json
import time
import os
from datetime import datetime

try:
    # colorama는 윈도우 콘솔에서 색을 잘 보여주기 위해 사용합니다.
    from colorama import init, Fore, Style
    init()
    COLOR_AVAILABLE = True
except Exception:
    # 색이 없다면 기본 출력으로 계속 동작합니다.
    class _Dummy:
        RESET_ALL = ""
        RED = ""
        YELLOW = ""
        GREEN = ""
        CYAN = ""
        MAGENTA = ""
        WHITE = ""
    Fore = _Dummy()
    Style = _Dummy()
    COLOR_AVAILABLE = False


class EmotionMonitor:
    """터미널에서 Elysia의 감정 상태를 실시간으로 보여주는 간단한 모니터.

    특징:
    - 사용자가 제안한 감정 레이블(PEACE, CURIOSITY, BOREDOM, MANIFESTATION)을 포함
    - 강도(intensity)에 따라 맥박처럼 펄스되는 표시
    - `elysia_state.json` 파일을 1초마다 읽음
    - colorama가 없을 때도 동작(무칼라)
    """

    def __init__(self):
        # 사용자 제공 매핑을 우선으로 포함
        self.emotion_map = {
            "peace": {"emoji": "💚", "label": "PEACE", "desc": "평화로운 상태... (zZz)", "color": Fore.GREEN},
            "curiosity": {"emoji": "💛", "label": "CURIOSITY", "desc": "반짝! 무언가 궁금해요! (*_*)", "color": Fore.YELLOW},
            "boredom": {"emoji": "🤍", "label": "BOREDOM", "desc": "조금... 심심해요... (-_-)", "color": Fore.WHITE},
            "manifestation": {"emoji": "✨", "label": "MANIFESTATION", "desc": "짜잔! 세상을 향해 손을 뻗는 중! (짠!)", "color": Fore.CYAN},
            # 이전에 있던 예비 상태들
            "happy": {"emoji": "😊", "label": "HAPPY", "desc": "기분 좋아요", "color": Fore.YELLOW},
            "sad": {"emoji": "😢", "label": "SAD", "desc": "슬퍼요", "color": Fore.BLUE},
            "neutral": {"emoji": "😐", "label": "NEUTRAL", "desc": "보통이에요", "color": Fore.WHITE},
        }

        # 애니메이션 프레임을 위한 내부 카운터
        self.frame = 0

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _format_intensity_bar(self, intensity, width=20, pulse=False):
        """강도 막대 생성. pulse=True면 프레임에 따라 약간의 애니메이션 적용."""
        filled = int(round(intensity * width))
        if pulse:
            # 프레임에 따라 살짝 흔들리게(펄스)
            pulse_offset = abs((self.frame % 6) - 3)  # 0,1,2,3,2,1,0...
            filled = max(0, min(width, filled + (pulse_offset - 1)))

        bar = "█" * filled + "░" * (width - filled)
        return bar

    def draw_frame(self, emotion_key, intensity, thought=None):
        self.clear_screen()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        info = self.emotion_map.get(emotion_key.lower(), self.emotion_map.get('neutral'))
        emoji = info['emoji']
        label = info['label']
        desc = info.get('desc', '')
        color = info.get('color', Fore.WHITE)

        # 펄스 애니메이션: intensity에 대해 pulse=True
        intensity_bar = self._format_intensity_bar(intensity, width=24, pulse=True)

        # 헤더
        print(f"\n{Fore.CYAN}=== Elysia Emotion Monitor ==={Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{now}{Style.RESET_ALL}\n")

        # 주요 상태
        print(f"{color}{emoji}  {label}{Style.RESET_ALL}  —  {desc}")
        print(f"Intensity: {color}{intensity_bar}{Style.RESET_ALL}  {intensity:.2f}/1.00")

        if thought:
            print(f"\nThought: {Fore.CYAN}{thought}{Style.RESET_ALL}")

        print('\n' + '=' * 50)

        # 다음 프레임을 위해 증가
        self.frame += 1

    def monitor_emotions(self, state_path='elysia_state.json', poll_interval=1.0):
        """루프: state 파일을 읽어 화면을 갱신합니다."""
        try:
            while True:
                try:
                    with open(state_path, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                except FileNotFoundError:
                    # 존재하지 않으면 기본값으로 표시
                    state = {'emotion': 'neutral', 'emotion_intensity': 0.2, 'current_thought': None}
                except json.JSONDecodeError:
                    state = {'emotion': 'neutral', 'emotion_intensity': 0.2, 'current_thought': None}

                emotion = state.get('emotion', 'neutral')
                # 일부 시스템은 'PEACE' 같은 대문자 키를 사용하므로 소문자/매핑을 유연하게 처리
                if isinstance(emotion, str):
                    emotion_key = emotion.strip().lower()
                else:
                    emotion_key = 'neutral'

                intensity = state.get('emotion_intensity', 0.5)
                try:
                    intensity = float(intensity)
                except Exception:
                    intensity = 0.5
                intensity = max(0.0, min(1.0, intensity))

                thought = state.get('current_thought')

                self.draw_frame(emotion_key, intensity, thought)

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}감정 모니터링을 종료합니다.{Style.RESET_ALL}")


if __name__ == '__main__':
    monitor = EmotionMonitor()
    monitor.monitor_emotions()