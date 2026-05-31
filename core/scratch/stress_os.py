import multiprocessing
import time
import sys

def cpu_burner():
    """무의미한 수학 연산을 반복하여 CPU에 극심한 병목(텐션)을 유발합니다."""
    while True:
        _ = 1000000 * 1000000

if __name__ == "__main__":
    print("=== [OS Stressor] Inducing Traffic Jam ===")
    print("Spawning CPU burning processes to create massive tension...")
    
    # 코어 수만큼 버너 실행하여 CPU 점유율 100% 유도
    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=cpu_burner)
        p.start()
        processes.append(p)
        
    try:
        time.sleep(10) # 10초간 병목 유발 후 자동 종료
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping OS Stressor...")
        for p in processes:
            p.terminate()
