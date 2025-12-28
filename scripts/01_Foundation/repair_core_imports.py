import os
import re

# Mapping for Core modules to their new 5-layer paths (Recursive Depth 3)
# Format: "Core.OldFolder" -> "Core.0X_Layer.0X_Subgroup.OldFolder"
core_mapping = {
    # 01_Foundation
    "Core.Foundation": "Core.01_Foundation.05_Foundation_Base.Foundation",
    "Core.Laws": "Core.01_Foundation.02_Legal_Ethics.Laws",
    "Core.Ethics": "Core.01_Foundation.02_Legal_Ethics.Ethics",
    "Core.Philosophy": "Core.01_Foundation.04_Philosophy.Philosophy",
    "Core.Security": "Core.01_Foundation.03_Security.Security",
    "Core.Elysia": "Core.01_Foundation.01_Core_Logic.Elysia",
    # 02_Intelligence
    "Core.Cognition": "Core.02_Intelligence.01_Reasoning.Cognition",
    "Core.Intelligence": "Core.02_Intelligence.01_Reasoning.Intelligence",
    "Core.Memory": "Core.02_Intelligence.02_Memory_Linguistics.Memory",
    "Core.Knowledge": "Core.02_Intelligence.02_Memory_Linguistics.Knowledge",
    "Core.Language": "Core.02_Intelligence.02_Memory_Linguistics.Language",
    "Core.Physics": "Core.02_Intelligence.03_Physics_Waves.Physics",
    "Core.Wave": "Core.02_Intelligence.03_Physics_Waves.Wave",
    "Core.Field": "Core.02_Intelligence.03_Physics_Waves.Field",
    "Core.Holographic": "Core.02_Intelligence.03_Physics_Waves.Holographic",
    "Core.Cortex": "Core.02_Intelligence.01_Reasoning.Cortex",
    "Core.Cognitive": "Core.02_Intelligence.01_Reasoning.Cognitive",
    "Core.Consciousness": "Core.02_Intelligence.04_Consciousness.Consciousness",
    "Core.Emotion": "Core.02_Intelligence.04_Consciousness.Emotion",
    "Core.Ether": "Core.02_Intelligence.04_Consciousness.Ether",
    "Core.Science": "Core.02_Intelligence.05_Science_Research.Science",
    # 03_Interaction
    "Core.Interface": "Core.03_Interaction.01_Interface.Interface",
    "Core.Sensory": "Core.03_Interaction.01_Interface.Sensory",
    "Core.Expression": "Core.03_Interaction.02_Expression.Expression",
    "Core.Communication": "Core.03_Interaction.02_Expression.Communication",
    "Core.Orchestra": "Core.03_Interaction.03_Coordination.Orchestra",
    "Core.Action": "Core.03_Interaction.03_Coordination.Action",
    "Core.Synesthesia": "Core.03_Interaction.03_Coordination.Synesthesia",
    "Core.Integration": "Core.03_Interaction.04_Network.Integration",
    "Core.Network": "Core.03_Interaction.04_Network.Network",
    "Core.Multimodal": "Core.03_Interaction.04_Network.Multimodal",
    "Core.Social": "Core.03_Interaction.04_Network.Social",
    "Core.Visual": "Core.03_Interaction.01_Interface.Visual",
    "Core.VR": "Core.03_Interaction.01_Interface.VR",
    # 04_Evolution
    "Core.Autonomy": "Core.04_Evolution.01_Growth.Autonomy",
    "Core.Evolution": "Core.04_Evolution.01_Growth.Evolution",
    "Core.Learning": "Core.04_Evolution.02_Learning.Learning",
    "Core.Creativity": "Core.04_Evolution.03_Creative.Creativity",
    "Core.Creation": "Core.04_Evolution.03_Creative.Creation",
    "Core.Studio": "Core.04_Evolution.03_Creative.Studio",
    # 05_Systems
    "Core.Monitor": "Core.05_Systems.01_Monitoring.Monitor",
    "Core.System": "Core.05_Systems.01_Monitoring.System",
    "Core.Demos": "Core.05_Systems.02_Simulation.Demos",
    "Core.World": "Core.05_Systems.02_Simulation.World",
    "Core.Time": "Core.05_Systems.02_Simulation.Time",
    "Core.Trinity": "Core.05_Systems.03_Existence.Trinity",
    "Core.AGI": "Core.05_Systems.03_Existence.AGI",
    "Core.Life": "Core.05_Systems.03_Existence.Life",
    "Core.Structure": "Core.05_Systems.01_Monitoring.Structure",
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
