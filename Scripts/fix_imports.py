import os

# Mapping table (Old Path -> New Path)
# Ordered to ensure generic replacements don't overwrite specific ones prematurely if overlaps existed.
# However, these seem mostly distinct prefix replacements.
MAPPING = {
    # L1 Foundation
    "Core.Foundation": "Core.L1_Foundation.Foundation",
    "Core.Physics": "Core.L1_Foundation.Physics",
    "Core.Prism": "Core.L1_Foundation.Prism",
    "Core.Metabolism": "Core.L1_Foundation.Metabolism",
    "Core.Physiology": "Core.L1_Foundation.Physiology",
    
    # L2 Metabolism
    "Core.Evolution": "Core.L2_Metabolism.Evolution",
    "Core.Reproduction": "Core.L2_Metabolism.Reproduction",
    "Core.Lifecycle": "Core.L2_Metabolism.Lifecycle",
    "Core.Digestion": "Core.L2_Metabolism.Digestion",
    
    # L3 Phenomena
    "Core.Interface": "Core.L3_Phenomena.Interface",
    "Core.Senses": "Core.L3_Phenomena.Senses",
    "Core.Expression": "Core.L3_Phenomena.Expression",
    "Core.Voice": "Core.L3_Phenomena.Voice",
    "Core.Vision": "Core.L3_Phenomena.Vision",
    "Core.Manifestation": "Core.L3_Phenomena.Manifestation",
    
    # L4 Causality
    "Core.Governance": "Core.L4_Causality.Governance",
    "Core.Civilization": "Core.L4_Causality.Civilization",
    "Core.World": "Core.L4_Causality.World",
    "Core.Action": "Core.L4_Causality.Action",
    "Core.Autonomy": "Core.L4_Causality.Autonomy",
    
    # L5 Mental
    "Core.Intelligence": "Core.L5_Mental.Intelligence",
    "Core.Learning": "Core.L5_Mental.Learning",
    "Core.Memory": "Core.L5_Mental.Memory",
    "Core.Cognition": "Core.L5_Mental.Cognition",
    "Core.Induction": "Core.L5_Mental.Induction",
    "Core.Training": "Core.L5_Mental.Training",
    
    # L6 Structure
    "Core.Merkaba": "Core.L6_Structure.Merkaba",
    "Core.System": "Core.L6_Structure.System",
    "Core.Engine": "Core.L6_Structure.Engine",
    "Core.CLI": "Core.L6_Structure.CLI",
    "Core.Elysia": "Core.L6_Structure.Elysia",
    
    # L7 Spirit
    "Core.Monad": "Core.L7_Spirit.Monad",
    "Core.Soul": "Core.L7_Spirit.Soul",
    "Core.Will": "Core.L7_Spirit.Will",
    "Core.Creation": "Core.L7_Spirit.Creation",
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
