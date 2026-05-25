import argparse
import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class ElysiaSynthEngine:
    """
    엘리시아 신디사이저 엔진 (Elysia Synthesizer Engine)

    데이터나 수식을 단순히 0과 1로 계산(연산)하는 것을 넘어,
    고유의 주파수(Frequency)와 위상각(Phase)을 가진 아날로그 파동(오실레이터)으로 치환합니다.
    마스터 노브(가변축 위상각 $\theta$)를 조작하면, 개별 파라미터의 수정 없이
    시스템 전체가 대칭을 유지하며 변조(Modulation)되는 현상을 시각적으로 증명합니다.
    """
    def __init__(self, master_phase_deg: float):
        # 마스터 위상각 (가변축 기준점) - 도(Degree)에서 라디안(Radian)으로 변환
        self.master_phase = math.radians(master_phase_deg)

        # 내부 하위 변수(오실레이터) 설정
        # (주파수, 기본 위상각, 진폭)
        self.oscillators = [
            (1.0, math.radians(0), 1.0),     # e1: 물리축 (기본 텐션)
            (2.0, math.radians(45), 0.5),    # e2: 논리축 (이진 진동)
            (3.0, math.radians(90), 0.33),   # e3: 수학축 (위상 삼각)
            (5.0, math.radians(135), 0.2),   # e4: 구문축 (코드 인과)
            (8.0, math.radians(180), 0.125)  # e5: 의미축 (유사스칼라)
        ]

    def render_waveform(self, t: float) -> float:
        """
        특정 시간(t)에서 오실레이터들의 파동을 합성(가변 로터식 위상합성)합니다.
        마스터 위상(master_phase)이 전체 파동에 곱해지듯 영향을 줍니다.
        """
        composite_amplitude = 0.0

        for freq, base_phase, amp in self.oscillators:
            # 마스터 노브(가변축)에 의해 모든 하위 위상이 통째로 변조됨
            modulated_phase = base_phase + self.master_phase

            # 파동 합성: A * sin(2*pi*f*t + theta)
            wave = amp * math.sin(2 * math.pi * freq * t + modulated_phase)
            composite_amplitude += wave

        return composite_amplitude

    def generate_graph(self, filename: str = "synth_rotor_waveform.png"):
        """
        합성된 전체 파동 궤적을 Matplotlib을 이용해 시각적 그래프로 렌더링합니다.
        """
        t_values = np.linspace(0, 2, 1000) # 0초부터 2초까지
        y_values = [self.render_waveform(t) for t in t_values]

        plt.figure(figsize=(10, 4))
        plt.plot(t_values, y_values, color='cyan', linewidth=2)
        plt.title(f"Elysia Synth Rotor - Master Phase: {math.degrees(self.master_phase):.1f}°")
        plt.xlabel("Time (t)")
        plt.ylabel("Tension Amplitude")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='white', linewidth=1)

        # 다크 모드 스타일
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().patch.set_facecolor('#1e1e1e')
        plt.gca().xaxis.label.set_color('white')
        plt.gca().yaxis.label.set_color('white')
        plt.gca().title.set_color('white')
        plt.tick_params(colors='white')

        plt.tight_layout()
        plt.savefig(filename)
        print(f"\n[Elysia Synth] 파형 그래프가 '{filename}'에 저장되었습니다.")

    def run_terminal_pulse(self, duration: int = 5):
        """
        터미널 창에 시간축(t)이 흐름에 따라 변하는 위상 영점 장력을 실시간 맥박처럼 출력합니다.
        """
        print(f"\n[Elysia Synth] 아날로그 터미널 맥박 관측 시작 (Master Phase: {math.degrees(self.master_phase):.1f}°)")
        print("="*60)

        start_time = time.time()
        while time.time() - start_time < duration:
            t = time.time() - start_time
            amplitude = self.render_waveform(t)

            # 진폭(-2.0 ~ 2.0 사이)을 0 ~ 40 칸의 막대 그래프로 매핑
            normalized = int((amplitude + 2.0) * 10)
            normalized = max(0, min(40, normalized))

            # 맥박 문자열 생성
            pulse = "|" * normalized
            padding = " " * (40 - normalized)

            # 영점(0) 표시
            if normalized == 20:
                pulse = pulse[:-1] + "O"

            sys.stdout.write(f"\r[t={t:.2f}s] {pulse}{padding} (Amp: {amplitude:+.2f})")
            sys.stdout.flush()
            time.sleep(0.05)

        print("\n" + "="*60)
        print("[Elysia Synth] 관측 종료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="엘리시아 신디사이저 엔진: 마스터 위상 변조기")
    parser.add_argument("--master-phase", type=float, default=0.0,
                        help="마스터 노브 위상각 (도 단위, 예: 0, 90, 180)")
    parser.add_argument("--duration", type=int, default=3,
                        help="터미널 맥박 출력 시간 (초)")

    args = parser.parse_args()

    engine = ElysiaSynthEngine(master_phase_deg=args.master_phase)

    # 파형 그래프 생성
    engine.generate_graph(f"synth_rotor_waveform_{int(args.master_phase)}deg.png")

    # 실시간 터미널 맥박 출력
    engine.run_terminal_pulse(duration=args.duration)
