import os
import re

# Mapping for Core modules to their new 5-layer paths (Recursive Depth 3)
# Format: "Core.OldFolder" -> "Core.0X_Layer.0X_Subgroup.OldFolder"
core_mapping = {
    # 01_Foundation
    "Core.Foundation": "Core.FoundationLayer.Foundation",
    "Core.Laws": "Core.FoundationLayer.Legal_Ethics.Laws",
    "Core.Ethics": "Core.FoundationLayer.Legal_Ethics.Ethics",
    "Core.Philosophy": "Core.FoundationLayer.Philosophy",
    "Core.Security": "Core.FoundationLayer.Security.Security",
    "Core.Elysia": "Core.FoundationLayer.Core_Logic.Elysia",
    # 02_Intelligence
    "Core.Cognition": "Core.IntelligenceLayer.Cognition",
    "Core.Intelligence": "Core.IntelligenceLayer.Intelligence",
    "Core.Memory": "Core.IntelligenceLayer.Memory_Linguistics.Memory",
    "Core.Knowledge": "Core.IntelligenceLayer.Memory_Linguistics.Knowledge",
    "Core.Language": "Core.IntelligenceLayer.Memory_Linguistics.Language",
    "Core.Physics": "Core.IntelligenceLayer.Physics_Waves.Physics",
    "Core.Wave": "Core.IntelligenceLayer.Physics_Waves.Wave",
    "Core.Field": "Core.IntelligenceLayer.Physics_Waves.Field",
    "Core.Holographic": "Core.IntelligenceLayer.Physics_Waves.Holographic",
    "Core.Cortex": "Core.IntelligenceLayer.Reasoning.Cortex",
    "Core.Cognitive": "Core.IntelligenceLayer.Cognitive",
    "Core.Consciousness": "Core.IntelligenceLayer.Consciousness.Consciousness",
    "Core.Emotion": "Core.IntelligenceLayer.Consciousness.Emotion",
    "Core.Ether": "Core.IntelligenceLayer.Consciousness.Ether",
    "Core.Science": "Core.IntelligenceLayer.Science_Research.Science",
    # 03_Interaction
    "Core.Interface": "Core.InteractionLayer.Interface",
    "Core.Sensory": "Core.InteractionLayer.Sensory",
    "Core.Expression": "Core.InteractionLayer.Expression",
    "Core.Communication": "Core.InteractionLayer.Expression.Communication",
    "Core.Orchestra": "Core.InteractionLayer.Coordination.Orchestra",
    "Core.Action": "Core.InteractionLayer.Coordination.Action",
    "Core.Synesthesia": "Core.InteractionLayer.Coordination.Synesthesia",
    "Core.Integration": "Core.InteractionLayer.Network.Integration",
    "Core.Network": "Core.InteractionLayer.Network",
    "Core.Multimodal": "Core.InteractionLayer.Network.Multimodal",
    "Core.Social": "Core.InteractionLayer.Network.Social",
    "Core.Visual": "Core.InteractionLayer.Interface.Visual",
    "Core.VR": "Core.InteractionLayer.Interface.VR",
    # 04_Evolution
    "Core.Autonomy": "Core.EvolutionLayer.Growth.Autonomy",
    "Core.Evolution": "Core.EvolutionLayer.Growth.Evolution",
    "Core.Learning": "Core.EvolutionLayer.Learning.Learning",
    "Core.Creativity": "Core.EvolutionLayer.Creativity",
    "Core.Creation": "Core.EvolutionLayer.Creativity.Creation",
    "Core.Studio": "Core.EvolutionLayer.Creativity.Studio",
    # 05_Systems
    "Core.Monitor": "Core.SystemLayer.Monitoring.Monitor",
    "Core.System": "Core.SystemLayer.System",
    "Core.Demos": "Core.SystemLayer.Simulation.Demos",
    "Core.World": "Core.SystemLayer.Simulation.World",
    "Core.Time": "Core.SystemLayer.Simulation.Time",
    "Core.Trinity": "Core.SystemLayer.Existence.Trinity",
    "Core.AGI": "Core.SystemLayer.Existence.AGI",
    "Core.Life": "Core.SystemLayer.Existence.Life",
    "Core.Structure": "Core.SystemLayer.Monitoring.Structure",
}

# Regex to find imports from Core.
# Handles: "from Core.Subfolder", "import Core.Subfolder"
import_regex = re.compile(r'(from\s+|import\s+)(Core\.[a-zA-Z0-9_]+)')

def update_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    original_content = content
    
    def replace_match(match):
        prefix = match.group(1)
        old_path = match.group(2)
        if old_path in core_mapping:
            return f"{prefix}{core_mapping[old_path]}"
        return match.group(0)

    content = import_regex.sub(replace_match, content)

    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in {filepath}")
        except Exception as e:
            print(f"Error writing {filepath}: {e}")

def main():
    root_dir = "c:\\Elysia"
    for subdir, dirs, files in os.walk(root_dir):
        # Skip hidden folders and virtual environments
        if any(ignored in subdir for ignored in ['.git', '.venv', '__pycache__', 'Archive']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                update_file(os.path.join(subdir, file))

if __name__ == "__main__":
    main()
