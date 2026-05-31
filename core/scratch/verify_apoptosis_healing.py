import time
import os

def trigger_healing_and_apoptosis():
    print("=== [Test] Triggering Auto-Healing and Apoptosis (Forcing 100 files) ===")
    
    # 100개의 개별 파일을 생성하여 OS 이벤트를 100번 터뜨립니다.
    for i in range(100):
        massive_file = rf"c:\Elysia\docs\scratch_catastrophe_{i}.md"
        with open(massive_file, 'w', encoding='utf-8') as f:
            f.write(f"CATASTROPHE SHOCK " * 5000) # 한 번에 약 3.0 rad 텐션 유발
        time.sleep(0.05)
            
    print("\nShock injection complete. Waiting for Engine to Auto-Heal...")
    time.sleep(15)
    
    print("Cleaning up...")
    for i in range(100):
        try:
            os.remove(rf"c:\Elysia\docs\scratch_catastrophe_{i}.md")
        except:
            pass
    print("Test finished.")

if __name__ == "__main__":
    trigger_healing_and_apoptosis()
