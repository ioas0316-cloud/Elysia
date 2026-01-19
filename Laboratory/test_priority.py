import psutil
import os
import time

def test_priority_boost():
    p = psutil.Process(os.getpid())
    print(f"Current Priority: {p.nice()}")
    
    try:
        print("Attempting to boost to HIGH_PRIORITY_CLASS...")
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print(f"New Priority: {p.nice()}")
        
        if p.nice() == psutil.HIGH_PRIORITY_CLASS:
            print("✅ SUCCESS: Windows allowed the priority boost.")
        else:
            print("❌ FAILURE: Priority did not change as expected.")
            
    except Exception as e:
        print(f"🚨 ERROR: {e}")

if __name__ == "__main__":
    test_priority_boost()
