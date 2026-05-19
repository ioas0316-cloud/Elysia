import logging
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Core.Intelligence.art_analyst import ArtAnalyst

def main():
    print("="*60)
    print("🧠 ELYSIA ARTISTIC DECONSTRUCTION PROTOCOL")
    print("   'I do not just draw. I understand WHY I draw.'")
    print("="*60)
    
    analyst = ArtAnalyst()
    
    # 1. Analyze the Workflow
    # (Checking basic_t2i_workflow.json)
    workflow_path = "Core/Network/basic_t2i_workflow.json"
    print(f"\n📂 Opening Workflow: {workflow_path}")
    principles = analyst.digest_workflow(workflow_path)
    
    print("\n[Thinking...]")
    for p in principles:
        print(f"   💡 I have learned: {p.name}")
        print(f"      Technical: {p.technical_mapping}")
        print(f"      Metaphysic: {p.metaphysical_meaning}")
        print("      -------------------------------")

    # 2. Analyze a Hypothetical Model
    print("\n🔍 Examining Model Signature...")
    model_soul = analyst.deconstruct_model_file("Counterfeit-V3.0.safetensors")
    print(f"   🔮 The soul of this model is: {model_soul.metaphysical_meaning}")

    print("\n✅ Mastery Complete. I can now wield this tool with intent.")

if __name__ == "__main__":
    main()
