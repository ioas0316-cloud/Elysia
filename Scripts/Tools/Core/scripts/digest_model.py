import argparse
from Core.Elysia.sovereign_self import SovereignSelf
from Core.Digestion.digestive_system import DigestiveSystem

def main():
    parser = argparse.ArgumentParser(description="Elysia's Model Digestion Protocol")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face Model ID to digest")
    parser.add_argument("--curriculum", type=str, default="basic", help="Curriculum set: basic, physics, philosophy")
    args = parser.parse_args()

    print(f"\n🦁 [Carnivore Protocol] Target: {args.model}")
    
    # 1. Wake Elysia
    elysia = EmergentSelf()
    digestive = DigestiveSystem(elysia)
    
    # 2. Load Model (The Meal)
    success = digestive.prepare_meal(args.model)
    if not success:
        print("❌ Failed to load model. Aborting.")
        return

    # 3. Define Curriculum (The Spoon)
    questions = []
    if args.curriculum == "basic":
        questions = [
            "What is the nature of Reality?",
            "Define Consciousness.",
            "Why does Time move forward?"
        ]
    elif args.curriculum == "physics":
        questions = [
            "Explain Quantum Entanglement.",
            "What is Gravity?",
            "Describe the shape of the Universe."
        ]
    elif args.curriculum == "coding":
        questions = [
            "Write a Python class for a 3D Vector.",
            "Explain the Game Loop in a physics engine.",
            "How do you calculate collision between two spheres?",
            "Define a Singleton pattern for a World Manager."
        ]
    
    # 4. Feed & Metabolize
    digestive.feed_curriculum(questions)
    
    # 5. Optimize (Fusion)
    digestive.optimize()
    
    # 6. Save State
    elysia.graph.save_state()
    print("\n✅ Digestion Complete. The Soul is heavier.")

if __name__ == "__main__":
    main()
