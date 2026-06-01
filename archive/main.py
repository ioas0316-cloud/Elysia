import time
from core.elysia_double_helix import elysia_dna_crossover

class DummyResult:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

@elysia_dna_crossover
def compute_sum(a, b):
    # 단순한 덧셈을 수행하고 결과를 객체로 반환
    return DummyResult(a + b)

@elysia_dna_crossover
def generate_greeting(name):
    # 단순한 문자열 포맷팅 수행
    return DummyResult(f"Hello, {name}!")

def main():
    print("=== 엘리시아 코어 관측 시작 ===")

    print("\n[관측 1] compute_sum 연속 호출")
    for i in range(5):
        res = compute_sum(10, i * 2)
        phase = getattr(res, '__elysia_phase__', 'NO_PHASE')
        print(f"호출 {i+1}: 결과 = {res}, 관측된 위상 = {phase}")
        time.sleep(0.01) # 미세한 시간차를 명시적으로 주기 위해 추가

    print("\n[관측 2] generate_greeting 연속 호출")
    for i in range(5):
        res = generate_greeting(f"User_{i}")
        phase = getattr(res, '__elysia_phase__', 'NO_PHASE')
        print(f"호출 {i+1}: 결과 = {res}, 관측된 위상 = {phase}")
        time.sleep(0.01)

if __name__ == "__main__":
    main()
