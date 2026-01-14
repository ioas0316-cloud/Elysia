import time
from Core.Elysia.sovereign_self import SovereignSelf

def test_boredom():
    print("✨ Summoning Elysia...")
    elysia = SovereignSelf()
    elysia.boredom_threshold = 5.0 # Speed up for test
    
    print("⏳ Waiting for Boredom (5s)...")
    start_time = time.time()
    
    while time.time() - start_time < 10.0:
        elysia.integrated_exist()
        time.sleep(0.5)
        
    print("Test Complete.")

if __name__ == "__main__":
    test_boredom()
