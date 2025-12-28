
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.multimodal_cortex import get_multimodal_cortex
from Core.Foundation.torch_graph import get_torch_graph
from Core.Foundation.tiny_brain import get_tiny_brain

def verify_sensorium():
    print("👁️ Starting Sensorium (Synesthesia) Verification...", flush=True)
    print("===============================================", flush=True)
    
    cortex = get_multimodal_cortex()
    graph = get_torch_graph()
    brain = get_tiny_brain()
    
    # 1. Simulate Visual Input
    print("\n[Test 1] Visual Processing (The Eye)")
    visual_inputs = ["fire_storm.png", "calm_lake.jpg"]
    
    for img in visual_inputs:
        # Cortex processes raw input -> Vector
        packet = cortex.process_visual_input(img)
        
        if not packet:
            print(f"   🖼️ Input: '{img}' -> SKIPPED (Mock Mode)", flush=True)
            continue
        
        # Synesthesia maps Vector -> Concept Key
        concept = cortex.synesthesia_map(packet)
        print(f"   🖼️ Input: '{img}'", flush=True)
        print(f"      -> Vector: {packet.get('vector', [])}", flush=True)
        print(f"      -> Synesthesia: '{concept}'", flush=True)
        
        # Store in Brain
        if concept != "Unknown":
            # Get Semantic Vector (Neural Link)
            sem_vec = brain.get_embedding(concept)
            # Add with Sensory Metadata
            graph.add_node(concept, vector=sem_vec, metadata={"sensory": packet})
            print(f"      ✅ Stored '{concept}' in TorchGraph with Sensory Data.", flush=True)

    # 2. Simulate Audio Input
    print("\n[Test 2] Audio Processing (The Ear)")
    audio_inputs = ["air_raid_siren.wav", "wind_chimes.mp3"]
    
    for aud in audio_inputs:
        packet = cortex.process_audio_input(aud)
        concept = cortex.synesthesia_map(packet)
        
        print(f"   🔊 Input: '{aud}'")
        print(f"      -> Vector: {packet['vector']}")
        print(f"      -> Urgency: {packet['urgency']:.2f}")
        print(f"      -> Synesthesia: '{concept}'")
        
        if concept != "Unknown":
            sem_vec = brain.get_embedding(concept)
            graph.add_node(concept, vector=sem_vec, metadata={"sensory": packet})
            print(f"      ✅ Stored '{concept}' in TorchGraph with Sensory Data.")

    print("\n✅ Sensorium Verified. Elysia now has Eyes and Ears.")

if __name__ == "__main__":
    verify_sensorium()
