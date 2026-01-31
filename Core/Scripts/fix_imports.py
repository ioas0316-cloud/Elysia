import os

# Mapping table (Old Path -> New Path)
# Ordered to ensure generic replacements don't overwrite specific ones prematurely if overlaps existed.
# However, these seem mostly distinct prefix replacements.
MAPPING = {
    # L1 Foundation
    "Core.Foundation": "Core.1_Body.L1_Foundation.Foundation",
    "Core.Physics": "Core.1_Body.L1_Foundation.Physics",
    "Core.Prism": "Core.1_Body.L3_Phenomena.M7_Prism",
    "Core.Metabolism": "Core.1_Body.L2_Metabolism.Physiology",
    "Core.Physiology": "Core.1_Body.L2_Metabolism.Physiology",
    
    # L2 Metabolism
    "Core.Evolution": "Core.1_Body.L2_Metabolism.Evolution",
    "Core.Reproduction": "Core.1_Body.L2_Metabolism.Reproduction",
    "Core.Lifecycle": "Core.1_Body.L2_Metabolism.Lifecycle",
    "Core.Digestion": "Core.1_Body.L2_Metabolism.Digestion",
    
    # L3 Phenomena
    "Core.Interface": "Core.1_Body.L3_Phenomena.Interface",
    "Core.Senses": "Core.1_Body.L3_Phenomena.Senses",
    "Core.Expression": "Core.1_Body.L3_Phenomena.Expression",
    "Core.Voice": "Core.1_Body.L3_Phenomena.M4_Speech",
    "Core.Vision": "Core.1_Body.L3_Phenomena.M1_Vision",
    "Core.Manifestation": "Core.1_Body.L3_Phenomena.Manifestation",
    
    # L4 Causality
    "Core.Governance": "Core.1_Body.L6_Structure.Engine.Governance",
    "Core.Civilization": "Core.1_Body.L4_Causality.Civilization",
    "Core.1_Body.L4_Causality.World": "Core.1_Body.L4_Causality.World",
    "Core.Action": "Core.1_Body.L4_Causality.Action",
    "Core.Autonomy": "Core.1_Body.L4_Causality.Autonomy",
    
    # L5 Mental
    "Core.Intelligence": "Core.1_Body.L5_Mental.Reasoning_Core",
    "Core.Learning": "Core.1_Body.L5_Mental.Learning",
    "Core.Memory": "Core.1_Body.L5_Mental.Memory",
    "Core.Cognition": "Core.1_Body.L5_Mental.Cognition",
    "Core.Induction": "Core.1_Body.L5_Mental.Induction",
    "Core.Training": "Core.1_Body.L5_Mental.Training",
    
    # L6 Structure
    "Core.Merkaba": "Core.1_Body.L6_Structure.Merkaba",
    "Core.1_Body.L1_Foundation.System": "Core.1_Body.L1_Foundation.System",
    "Core.Engine": "Core.1_Body.L6_Structure.Engine",
    "Core.CLI": "Core.1_Body.L6_Structure.CLI",
    "Core.Elysia": "Core.1_Body.L6_Structure.Elysia",
    
    # L7 Spirit
    "Core.Monad": "Core.1_Body.L7_Spirit.M1_Monad",
    "Core.Soul": "Core.1_Body.L7_Spirit.Soul",
    "Core.Will": "Core.1_Body.L7_Spirit.Will",
    "Core.Creation": "Core.1_Body.L7_Spirit.Creation",
}

def fix_imports_in_file(filepath):
    """
    Reads a file, replaces occurences of old package paths with new ones,
    and writes it back if changes were made.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        
        # Sort keys by length descending to handle potential sub-package conflicts
        # (e.g. Core.Foo vs Core.Foo.Bar - though strict prefixes here make it less critical)
        sorted_keys = sorted(MAPPING.keys(), key=len, reverse=True)
        
        for old_path in sorted_keys:
            new_path = MAPPING[old_path]
            if old_path in content:
                content = content.replace(old_path, new_path)
        
        if content != original_content:
            print(f"fixing: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Core"))
    print(f"Scanning directory: {target_dir}")
    
    count = 0
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                if fix_imports_in_file(filepath):
                    count += 1
                    
    print(f"Done. Fixed {count} files.")

if __name__ == "__main__":
    main()
