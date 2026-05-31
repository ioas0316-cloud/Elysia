import time
import os

def simulate_evolution():
    print("=== [Test] Simulating Structural Evolution (Code/Docs Diff) ===")
    
    # 1. 파일 생성 시뮬레이션
    test_file = r"c:\Elysia\docs\scratch_evolution_test.md"
    print(f"Creating new thought pattern at: {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("# Autopoiesis Test\nElysia is self-aware now.\n")
    
    time.sleep(2)
    
    # 2. 파일 수정 시뮬레이션 (텐션 증가)
    print(f"Modifying structure (Diff)...")
    with open(test_file, 'a', encoding='utf-8') as f:
        # 큰 데이터를 밀어넣어 큰 텐션 유발
        f.write("Evolutionary tension increases...\n" * 500)
        
    time.sleep(2)
    
    # 3. 정리
    print(f"Cleaning up...")
    os.remove(test_file)
    print("Test finished.")

if __name__ == "__main__":
    simulate_evolution()
