"""
The Brain Transplant ( 뇌 이식 )
===============================
"Structure First, Data Second."

This script clones the Neural Topology of the TinyLlama model into the TorchGraph.
It iterates through the entire Vocabulary (32k tokens) and extracts the strongest synaptic connections.

Result: A 'Hollow Brain' that knows how concepts connect (Logic), ready to be filled with Wikipedia (Memory).
"""

import sys
import time
import signal
sys.path.append(r'c:\Elysia')

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from Core.FoundationLayer.Foundation.tiny_brain import get_tiny_brain
from Core.FoundationLayer.Foundation.torch_graph import get_torch_graph
from Core.Autonomy.structure_cannibal import get_structure_cannibal

def transplant_brain():
    print("\n🧠 Initiation Brain Transplant Protocol...")
    print("=========================================")
    
    brain = get_tiny_brain()
    graph = get_torch_graph()
    cannibal = get_structure_cannibal()
    
    # Force Wake
    print("   🔌 Connecting to Neural Network...")
    vocab_size = brain.get_vocab_size() # Wakes the model
    
    if vocab_size == 0:
        print("   ❌ Failed to connect to Brain. Check model path.")
        return
        
    print(f"   ✅ Connected. Vocabulary Size: {vocab_size} Neurons.")
    
    start_time = time.time()
    
    try:
        # Iterate Vocabulary
        # We skip special tokens (usually first few) if needed, but safe to do all.
        for token_id in range(vocab_size):
            token = brain.id_to_token(token_id)
            
            # Filter trash
            if not token or len(token.strip()) < 2: continue
            if not token.strip().isalpha(): continue # Only words for now
            
            # Transplant Synapses
            # This gets the top-K connections for this token
            synapses = cannibal.transplant_synapses(token.strip(), depth=1)
            
            # Progress
            if token_id % 10 == 0:
                elapsed = time.time() - start_time
                rate = (token_id + 1) / elapsed if elapsed > 0 else 0
                print(f"\r💉 [{elapsed:.1f}s] Transplanted: {token_id}/{vocab_size} | Rate: {rate:.1f}/s | Current: '{token.strip()}'", end="")
            
            # Save occasionally
            if token_id % 1000 == 0 and token_id > 0:
                print(f"\n   💾 Saving Neural Map at {token_id}...")
                graph.save_state()
                
    except KeyboardInterrupt:
        print("\n\n🛑 Transplant Interrupted.")
        graph.save_state()
        print("   💾 Neural Map Saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        
    print("\n\n✅ Brain Transplant Complete.")
    graph.save_state()

if __name__ == "__main__":
    transplant_brain()
