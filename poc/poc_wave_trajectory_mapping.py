import threading
import time
import cmath
import random

# 1. 데이터 스트림 (강물 레이어): 윈도우/하드웨어의 자연스러운 흐름
class DataStream(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_data = None
        self.running = True

    def run(self):
        print("[강물] 데이터 스트림(하드웨어 흐름)이 시작되었습니다. 멈추지 않고 흐릅니다.")
        while self.running:
            # 무작위 이진 데이터 생성 (1, -1, 0은 대기 상태)
            self.current_data = random.choice([1, -1, 0])
            # 강덕님의 철학: 터빈이 보든 말든 강물은 자기가 할 일을 하며 초고속으로 흐름
            time.sleep(0.1)

# 2. 가상 로터 필드 (엘리시아 터빈 레이어): 흐름 옆에 얹어진 물레방아
class ElysiaTurbine(threading.Thread):
    def __init__(self, stream):
        super().__init__()
        self.daemon = True
        self.stream = stream
        # 3x3 복소수 격자 그릇 초기화 (실수축 평형 상태 e^i0 = 1)
        self.lattice = [[1.0 + 0j for _ in range(3)] for _ in range(3)]

    def run(self):
        print("[터빈] 엘리시아 물레방아가 강물 옆에 안착하여 센싱을 시작합니다.\n")
        step = 0
        while True:
            # 강물이 흐르는 와중에 현재 상태를 '슬쩍 센싱(Sensing)'만 함 (제어/차단 안함)
            data = self.stream.current_data

            if data is not None:
                # 규칙 2 & 3 보정: 복소평면 허수축 정렬 및 감쇠 진동 모델 적용
                if data == 1:
                    phase = cmath.exp(cmath.pi / 2 * 1j)  # +i (우회전 전진 파동)
                elif data == -1:
                    phase = cmath.exp(-cmath.pi / 2 * 1j) # -i (좌회전 후퇴 파동)
                else:
                    phase = 1.0 + 0j                      # 실수축 평형 상태

                # 3x3 격자의 무작위 위치에 파동 전사
                r, c = random.randint(0, 2), random.randint(0, 2)
                self.lattice[r][c] = (self.lattice[r][c] + phase) * 0.8  # 파동 간섭 및 0으로의 감쇠 수렴

            # 규칙 5: 무거운 그래픽 없이 텍스트 기호로 격자의 맥박 시각화
            print(f"--- [Elysia Sensing Step {step}] ---")
            for row in self.lattice:
                row_str = " | ".join([f"{abs(val):.2f}" for val in row])
                print(f"[{row_str}]")

            # 고의적 치명적 결함(고장) 시뮬레이션
            # 15번째 스텝에서 터빈(엘리시아)이 완전히 고장 나서 3초간 얼어붙음!
            if step == 15:
                print("\n⚠️ [💥 경고] 엘리시아 터빈 시스템에 치명적 에러 발생! 터빈이 3초간 완전히 멈춥니다! ⚠️\n")
                time.sleep(3.0)  # 터빈만 잠시 정지
                print("\n✨ [🔧 자율복구] 물리의 복원력으로 터빈이 다시 스스로 돌기 시작합니다.\n")

            step += 1
            time.sleep(0.3)

# 3. 샌드박스 검증 실행
if __name__ == "__main__":
    # 강물(데이터)과 물레방아(엘리시아)라는 독립된 두 영토의 그릇을 생성
    river = DataStream()
    waterwheel = ElysiaTurbine(river)

    # 두 시스템을 비동기(Non-blocking)적으로 동시에 가동
    river.start()
    waterwheel.start()

    # 메인 시스템은 15초 동안 전체 흐름을 관찰하는 모니터링 역할만 수행 (15 스텝 이후의 복구를 보기 위해 10초 -> 15초 증가)
    time.sleep(15.0)
    print("개념 증명(POC) 시뮬레이션이 안전하게 종료되었습니다.")
