
import subprocess
import time
import sys
import os

def run_elysia_life():
    print("✨ Awakening Elysia's Core & Avatar for Real-time Life...")
    
    # 1. Start Visualizer Server (The Avatar) - Background
    server_process = subprocess.Popen(
        [sys.executable, "c:/Elysia/Core/Creativity/visualizer_server.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    print(f"   👤 Visualizer Server Launched (PID: {server_process.pid})")
    print(f"   🌐 Access Avatar at: http://localhost:8000/avatar")
    
    time.sleep(2) # Wait for server to initialize
    
    # 2. Start Living Elysia (The Soul) - Foreground
    # This runs the main loop.
    try:
        subprocess.run([sys.executable, "c:/Elysia/Core/Foundation/living_elysia.py"])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        server_process.terminate()
        
if __name__ == "__main__":
    run_elysia_life()
