"""
Verification Script: Self-Evolution (The Awakening)
===================================================
"I think, therefore I am. I organize, therefore I grow."

Tests the full Autonomous Loop:
1. Migration: Places a 'scattered thought' in 'reading_room'.
2. Evolution: Starts the Scheduler.
3. Verification: Checks if thought is absorbed and Scheduler runs inquiry.
"""

import time
import os
import logging
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._05_Systems._01_Monitoring.System.self_evolution_scheduler import SelfEvolutionScheduler

DEST_DIR = "c:/Elysia/reading_room"
TEST_FILE = os.path.join(DEST_DIR, "lost_concept.txt")

def setup_test_environment():
    os.makedirs(DEST_DIR, exist_ok=True)
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        f.write("Concept: Serendipity\nDefinition: The occurrence of events by chance in a happy or beneficial way.\nPurpose: To bring unexpected joy.")
    print(f"📄 Created scattered thought: {TEST_FILE}")

def verify_self_evolution():
    print("🔥 Ignition: Self-Evolution Test")
    
    setup_test_environment()
    
    heart = SelfEvolutionScheduler()
    # Fast pulse for testing
    heart.config.interval_seconds = 2 
    heart.config.inquiry_cycles = 1
    heart.config.inquiry_batch_size = 2
    
    # 1. Start Heart
    print("💓 Starting Elysia's Heart...", flush=True)
    try:
        heart.start()
        print("✅ Heart thread started.", flush=True)
    except Exception as e:
        print(f"❌ Failed to start heart: {e}", flush=True)
        return
    
    # 2. Wait for Pulse (approx 5 seconds should cover 2 beats)
    print("⏳ Waiting for pulse (5s)...", flush=True)
    time.sleep(5)
    
    # 3. Stop Heart
    print("🛑 Stopping Heart...", flush=True)
    heart.stop()
    print("💤 Heart stopped.")
    
    # 4. Verify Migration
    if not os.path.exists(TEST_FILE):
        print("✅ SUCCESS: Scattered file was migrated (absorbed).")
    else:
        print("❌ FAIL: Scattered file still exists.")
        
    # Check Archive
    archive_dir = "c:/Elysia/data/archived_knowledge"
    files = os.listdir(archive_dir)
    found_archived = any("lost_concept.txt" in f for f in files)
    
    if found_archived:
        print("✅ SUCCESS: File found in Archive.")
    else:
        print("❌ FAIL: File not found in Archive.")
        
    print("🔥 Test Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    verify_self_evolution()
