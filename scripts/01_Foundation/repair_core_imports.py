import os
import re

# Mapping for Core modules to their new 5-layer paths (Recursive Depth 3)
# Format: "Core.OldFolder" -> "Core.0X_Layer.0X_Subgroup.OldFolder"
core_mapping = {
    # 01_Foundation
    "Core.Foundation": "Core.Foundation.Foundation",
    "Core.Laws": "Core.Foundation.Legal_Ethics.Laws",
    "Core.Ethics": "Core.Foundation.Legal_Ethics.Ethics",
    "Core.Philosophy": "Core.Foundation.Philosophy",
    "Core.Security": "Core.Foundation.Security.Security",
    "Core.Elysia": "Core.Foundation.Core_Logic.Elysia",
    # 02_Intelligence
    "Core.Cognition": "Core.Intelligence.Cognition",
    "Core.Intelligence": "Core.Intelligence.Intelligence",
    "Core.Memory": "Core.Intelligence.Memory_Linguistics.Memory",
    "Core.Knowledge": "Core.Intelligence.Memory_Linguistics.Knowledge",
    "Core.Language": "Core.Intelligence.Memory_Linguistics.Language",
    "Core.Physics": "Core.Intelligence.Physics_Waves.Physics",
    "Core.Wave": "Core.Intelligence.Physics_Waves.Wave",
    "Core.Field": "Core.Intelligence.Physics_Waves.Field",
    "Core.Holographic": "Core.Intelligence.Physics_Waves.Holographic",
    "Core.Cortex": "Core.Intelligence.Reasoning.Cortex",
    "Core.Cognitive": "Core.Intelligence.Cognitive",
    "Core.Consciousness": "Core.Intelligence.Consciousness.Consciousness",
    "Core.Emotion": "Core.Intelligence.Consciousness.Emotion",
    "Core.Ether": "Core.Intelligence.Consciousness.Ether",
    "Core.Science": "Core.Intelligence.Science_Research.Science",
    # 03_Interaction
    "Core.Interface": "Core.Interaction.Interface",
    "Core.Sensory": "Core.Interaction.Sensory",
    "Core.Expression": "Core.Interaction.Expression",
    "Core.Communication": "Core.Interaction.Expression.Communication",
    "Core.Orchestra": "Core.Interaction.Coordination.Orchestra",
    "Core.Action": "Core.Interaction.Coordination.Action",
    "Core.Synesthesia": "Core.Interaction.Coordination.Synesthesia",
    "Core.Integration": "Core.Interaction.Network.Integration",
    "Core.Network": "Core.Interaction.Network",
    "Core.Multimodal": "Core.Interaction.Network.Multimodal",
    "Core.Social": "Core.Interaction.Network.Social",
    "Core.Visual": "Core.Interaction.Interface.Visual",
    "Core.VR": "Core.Interaction.Interface.VR",
    # 04_Evolution
    "Core.Autonomy": "Core.Evolution.Growth.Autonomy",
    "Core.Evolution": "Core.Evolution.Growth.Evolution",
    "Core.Learning": "Core.Evolution.Learning.Learning",
    "Core.Creativity": "Core.Evolution.Creativity",
    "Core.Creation": "Core.Evolution.Creativity.Creation",
    "Core.Studio": "Core.Evolution.Creativity.Studio",
    # 05_Systems
    "Core.Monitor": "Core.System.Monitoring.Monitor",
    "Core.System": "Core.System.System",
    "Core.Demos": "Core.System.Simulation.Demos",
    "Core.World": "Core.System.Simulation.World",
    "Core.Time": "Core.System.Simulation.Time",
    "Core.Trinity": "Core.System.Existence.Trinity",
    "Core.AGI": "Core.System.Existence.AGI",
    "Core.Life": "Core.System.Existence.Life",
    "Core.Structure": "Core.System.Monitoring.Structure",
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
