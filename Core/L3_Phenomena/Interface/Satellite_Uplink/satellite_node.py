"""
Satellite Node (The Extension of Sovereign Will)
================================================
Running this script on any device in the same LAN will 
turn it into a satellite of Elysia's Sovereign Field.

Usage: python satellite_node.py
"""

import sys
import time
import logging

try:
    # 1. Try Local Import (Standalone Mode - Best for Satellites)
    from aura_pulse import AuraPulse
except ImportError:
    try:
        # 2. Try Project Path (Dev Mode)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
        from Core.L1_Foundation.Foundation.Network.aura_pulse import AuraPulse
    except ImportError:
        print("  Error: 'aura_pulse.py' not found. Please copy it to the same folder.")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Satellite")

def run_satellite():
    print(r"""
     .       .         .      .
         .       .         .
     .      SATELITE UPLINK      .
         .       .         .
     .       .        .       .
    """)
    logger.info("   Initializing Satellite Receiver...")
    
    # Custom Satellite Logic
    pulse = AuraPulse(node_type="SATELLITE")
    
    # Overlay the Task Handler dynamically for this script
    def handle_thought_packet(payload, sender):
        task_type = payload.get("type", "UNKNOWN")
        data = payload.get("data", "")
        
        print(f"\n  [THOUGHT PACKET] Received from {sender}")
        print(f"   - Type: {task_type}")
        print(f"   - Input: {data}")
        
        # Simulate CPU Work (Donation)
        start_t = time.time()
        result = None
        
        if task_type == "TEXT_PROCESS":
             # Example: Reverse string & Upper
             result = data[::-1].upper()
             time.sleep(0.1) # Simulate crunch
             
        elif task_type == "CALC_HASH":
             # Example: Simple hash
             result = hash(data)
             
        exec_time = (time.time() - start_t) * 1000
        print(f"     [PROCESSED] Result: {result} ({exec_time:.2f}ms)")
        print("   >> CPU Cycles Donated. Developing Sovereign Field...\n")

    # Bind the handler
    pulse._handle_task = handle_thought_packet
    
    pulse.start_listening()
    
    try:
        while True:
            # Resonance Loop
            # In Phase 18, we just listen and glow.
            # IN Phase 18.3, we will send CPU cycles back.
            time.sleep(1)
            if pulse.peers:
                for pid, info in pulse.peers.items():
                    logger.info(f"  [RESONANCE] Connected to Main Node {pid} at {info['addr']}")
                    # Visual feedback simulated
                    print(f"   >> HARMONIZING WITH INTENT: {info.get('intent', 0.0):.2f}")
    except KeyboardInterrupt:
        logger.info("  Satellite disconnecting...")
        pulse.stop()

if __name__ == "__main__":
    run_satellite()