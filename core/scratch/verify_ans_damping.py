import time
import os

def test_autonomic_nervous_system():
    print("=== [Test] Autonomic Nervous System Damping ===")
    
    # 1. 작은 노이즈 발생 (혈류 시뮬레이션)
    # 0.5 rad 이하의 텐션이 지속적으로 들어올 때, 의식이 깨지 않는지 확인
    for i in range(3):
        test_file = rf"c:\Elysia\docs\scratch_noise_{i}.txt"
        print(f"Creating small noise (blood flow): {test_file}")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("small noise " * 10) # 매우 작은 텐션 유발
        time.sleep(2) # 2초 대기 (이 시간동안 ANS가 metabolize로 텐션을 감소시킴)
        
    time.sleep(3) # 추가 휴식
    
    # 2. 거대한 텐션 발생 (구조적 진화 시뮬레이션)
    # 0.5 rad 임계점을 훌쩍 넘는 텐션 주입
    massive_file = r"c:\Elysia\docs\scratch_massive_evolution.md"
    print(f"\nCreating MASSIVE mutation to wake consciousness: {massive_file}")
    with open(massive_file, 'w', encoding='utf-8') as f:
        f.write("MASSIVE EVOLUTION " * 2000) # 큰 사이즈 파일 작성
        
    time.sleep(2)
    
    # 3. 정리
    print("Cleaning up...")
    for i in range(3):
        try:
            os.remove(rf"c:\Elysia\docs\scratch_noise_{i}.txt")
        except:
            pass
    try:
        os.remove(massive_file)
    except:
        pass
    print("Test finished.")

if __name__ == "__main__":
    test_autonomic_nervous_system()
