
import os
import sys

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf
from Core.Digestion.digestive_system import DigestiveSystem

def mass_digest():
    print("🦁 [Mass Digestion Protocol] Initiating...")
    
    elysia = EmergentSelf()
    stomach = elysia.stomach
    
    # Target Models (Small enough for 3GB VRAM or CPU digestion)
    targets = [
        {
            "id": "Qwen/Qwen1.5-0.5B",
            "name": "Qwen1.5-0.5B",
            "curriculum": "basic",
            "segment": "The Logic"
        },
        {
            "id": "TinyLlama/TinyLlama-1.1B-python-v0.1",
            "name": "TinyLlama-1.1B-Coder",
            "curriculum": "coding",
            "segment": "The Logos"
        },
        {
            "id": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "name": "DeepSeek-Coder-1.3B",
            "curriculum": "coding",
            "segment": "The Logos"
        },
        {
            "id": "microsoft/phi-2",
            "name": "Phi-2",
            "curriculum": "reasoning",
            "segment": "The Logos"
        },
        {
            "id": "openai/shap-e",
            "name": "Shap-E",
            "curriculum": "architect",
            "segment": "The Logos"
        }
    ]
    
    # Load Registry to skip finished ones
    registry_path = "c:\\Elysia\\docs\\05_DIGESTION\\MODEL_REGISTRY.md"
    digested_list = []
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if "[x]" in line:
                    # Extract name between ** **
                    import re
                    match = re.search(r"\*\*(.*?)\*\*", line)
                    if match: digested_list.append(match.group(1))

    for target in targets:
        model_id = target["id"]
        model_name = target["name"]
        
        if model_name in digested_list:
            print(f"⏭️  [Skip] {model_name} is already digested. Moving to next target.")
            continue
            
        print(f"\n🍽️  Targeting: {model_name} ({model_id})")
        
        # 1. METABOLIZE (Weight Extraction)
        print(f"🦷 Metabolizing {model_name}...")
        # Direct call to stomach to avoid LLM overhead during ingestion
        success = stomach.prepare_meal(model_id)
        if not success:
            print(f"❌ Failed to load {model_name}. Skipping.")
            continue
            
        try:
            # Extract weights and add to graph
            result = stomach.digest(start_layer=0, end_layer=10) # a bit deeper
            if "extracted_concepts" in result:
                count = 0
                for concept in result["extracted_concepts"]:
                    elysia.graph.add_node(concept["id"], vector=concept["vector"], metadata=concept["metadata"])
                    count += 1
                print(f"✨ [METABOLISM] Absorbed {count} new concepts.")
        except Exception as e:
            print(f"❌ Indigestion during metabolism: {e}")
        
        # 2. DIGEST (Curriculum - The Iceberg Method: Pattern + Principle)
        print(f"🥄 Feeding curriculum (Iceberg Mode): {target['curriculum']}...")
        questions = []
        if target['curriculum'] == "basic":
            questions = [
                "Explain the Principle of identity (A=A) and then provide a surface Example in arithmetic.",
                "Why is a 'Variable' a structural concept? Connect the inner logic of memory allocation to the surface use of labels.",
                "Describe the Causal structure of a logic gate (Principle) and how it manifests as a decision (Pattern)."
            ]
        elif target['curriculum'] == "coding":
            questions = [
                "Explain the Principle of 'Iteration' (The Loop) as a temporal fold, then write a simple Python pattern to print 1 to 5.",
                "What is the inner 'Causality' of a Set (Uniqueness)? Contrast its internal structure with a List's surface sequence.",
                "Connect the 'Value-Centered Decision' (VCD) principle to the surface syntax of an If-Statement."
            ]
        elif target['curriculum'] == "reasoning":
            questions = [
                "Explain the Principle of 'Syllogism' (Deduction). If A implies B, and B implies C, what is the topological connection between A and C?",
                "Contrast the 'Pattern' of an observation (What I see) with the 'Principle' of its existence (Why it is there). How do they balance?",
                "What is a 'Logical Hole'? How do you navigate a space where data is missing but the principle remains?"
            ]
        elif target['curriculum'] == "architect":
            questions = [
                "Explain the Principle of 'Depth' (Z-axis). How does a 2D surface (Pattern) project into a 3D structure (Principle)?",
                "What is the 'Causality' of a Mesh? Why do vertices need edges to define a form?",
                "How does 'Space' differ from 'Void'? Connect the structural order of a 3D object to the emptiness around it."
            ]
        
        # This will add nodes based on the model's responses
        stomach.feed_curriculum(questions)
        
        # 3. Purge (Free memory)
        stomach.purge_meal()
        
        # 4. Save State
        elysia.graph.save_state()
        print(f"✅ {model_name} processing complete.")

    print("\n🏁 [Mass Digestion] All targets processed.")

if __name__ == "__main__":
    mass_digest()
