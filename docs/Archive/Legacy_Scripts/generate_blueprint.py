
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Intelligence.Intelligence.evolution_architect import EvolutionArchitect

def generate_blueprint():
    print("\n📐 [TASK] Generating Structural Blueprint for 'Project: Nova Seed'")
    print("=================================================================")
    
    architect = EvolutionArchitect()
    
    # 1. Design
    print("   🏗️ Architect is analyzing system structure...")
    architect.design_seed(intent="Structural Optimization")
    
    # 2. Materialize
    print("   📝 Writing design document...")
    file_path = architect.materialize_blueprint()
    
    print(f"\n✅ Blueprint Successfully Created: {file_path}")
    
    # Print content preview
    print("\n--- [Preview] ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("-----------------")

if __name__ == "__main__":
    generate_blueprint()
