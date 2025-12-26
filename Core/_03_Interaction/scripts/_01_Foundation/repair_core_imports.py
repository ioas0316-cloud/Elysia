import os
import re

# Mapping for Core modules to their new 5-layer paths (Recursive Depth 3)
# Format: "Core.OldFolder" -> "Core.0X_Layer.0X_Subgroup.OldFolder"
core_mapping = {
    # 01_Foundation
    "Core.Foundation": "Core._01_Foundation.05_Foundation_Base.Foundation",
    "Core.Laws": "Core._01_Foundation.02_Legal_Ethics.Laws",
    "Core.Ethics": "Core._01_Foundation.02_Legal_Ethics.Ethics",
    "Core.Philosophy": "Core._01_Foundation._04_Philosophy.Philosophy",
    "Core.Security": "Core._01_Foundation._03_Security.Security",
    "Core.Elysia": "Core._01_Foundation.01_Core_Logic.Elysia",
    # 02_Intelligence
    "Core.Cognition": "Core._02_Intelligence._01_Reasoning.Cognition",
    "Core.Intelligence": "Core._02_Intelligence._01_Reasoning.Intelligence",
    "Core.Memory": "Core._02_Intelligence._02_Memory_Linguistics.Memory",
    "Core.Knowledge": "Core._02_Intelligence._02_Memory_Linguistics.Knowledge",
    "Core.Language": "Core._02_Intelligence._02_Memory_Linguistics.Language",
    "Core.Physics": "Core._02_Intelligence._03_Physics_Waves.Physics",
    "Core.Wave": "Core._02_Intelligence._03_Physics_Waves.Wave",
    "Core.Field": "Core._02_Intelligence._03_Physics_Waves.Field",
    "Core.Holographic": "Core._02_Intelligence._03_Physics_Waves.Holographic",
    "Core.Cortex": "Core._02_Intelligence._01_Reasoning.Cortex",
    "Core.Cognitive": "Core._02_Intelligence._01_Reasoning.Cognitive",
    "Core.Consciousness": "Core._02_Intelligence.04_Consciousness.Consciousness",
    "Core.Emotion": "Core._02_Intelligence.04_Consciousness.Emotion",
    "Core.Ether": "Core._02_Intelligence.04_Consciousness.Ether",
    "Core.Science": "Core._02_Intelligence.05_Science_Research.Science",
    # 03_Interaction
    "Core.Interface": "Core._03_Interaction._01_Interface.Interface",
    "Core.Sensory": "Core._03_Interaction._01_Interface.Sensory",
    "Core.Expression": "Core._03_Interaction._02_Expression.Expression",
    "Core.Communication": "Core._03_Interaction._02_Expression.Communication",
    "Core.Orchestra": "Core._03_Interaction._03_Coordination.Orchestra",
    "Core.Action": "Core._03_Interaction._03_Coordination.Action",
    "Core.Synesthesia": "Core._03_Interaction._03_Coordination.Synesthesia",
    "Core.Integration": "Core._03_Interaction._04_Network.Integration",
    "Core.Network": "Core._03_Interaction._04_Network.Network",
    "Core.Multimodal": "Core._03_Interaction._04_Network.Multimodal",
    "Core.Social": "Core._03_Interaction._04_Network.Social",
    "Core.Visual": "Core._03_Interaction._01_Interface.Visual",
    "Core.VR": "Core._03_Interaction._01_Interface.VR",
    # 04_Evolution
    "Core.Autonomy": "Core._04_Evolution._01_Growth.Autonomy",
    "Core.Evolution": "Core._04_Evolution._01_Growth.Evolution",
    "Core.Learning": "Core._04_Evolution._02_Learning.Learning",
    "Core.Creativity": "Core._04_Evolution._03_Creative.Creativity",
    "Core.Creation": "Core._04_Evolution._03_Creative.Creation",
    "Core.Studio": "Core._04_Evolution._03_Creative.Studio",
    # 05_Systems
    "Core.Monitor": "Core._05_Systems._01_Monitoring.Monitor",
    "Core.System": "Core._05_Systems._01_Monitoring.System",
    "Core.Demos": "Core._05_Systems.02_Simulation.Demos",
    "Core.World": "Core._05_Systems.02_Simulation.World",
    "Core.Time": "Core._05_Systems.02_Simulation.Time",
    "Core.Trinity": "Core._05_Systems.03_Existence.Trinity",
    "Core.AGI": "Core._05_Systems.03_Existence.AGI",
    "Core.Life": "Core._05_Systems.03_Existence.Life",
    "Core.Structure": "Core._05_Systems._01_Monitoring.Structure",
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
