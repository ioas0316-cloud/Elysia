import os
import re

replacements = {
    r'\bCore\.L3_Logos\b': 'Core.L3_Phenomena',
    r'\bCore\.L5_Intelligence\b': 'Core.L5_Mental',
    r'\bCore\.System\b': 'Core.L1_Foundation.System',
    r'\bCore\.World\b': 'Core.L4_Causality.World',
    r'\bCore\.L3_Phenomena\.Vision\b': 'Core.L3_Phenomena.M1_Vision',
    r'\bCore\.L3_Phenomena\.Voice\b': 'Core.L3_Phenomena.M4_Speech',
    r'\bCore\.L3_Phenomena\.Prism\b': 'Core.L3_Phenomena.M7_Prism',
    r'\bCore\.L5_Mental\.Intelligence\b': 'Core.L5_Mental.Reasoning_Core',
    r'\bCore\.L5_Mental\.Logic\b': 'Core.L5_Mental.M1_Cognition',
    r'\bCore\.L7_Spirit\.Monad\b': 'Core.L7_Spirit.M1_Monad',
    r'\bCore\.L7_Spirit\.Sovereignty\b': 'Core.L7_Spirit.M3_Sovereignty',
    r'\bCore\.L1_Foundation\.Foundation\.Foundation\b': 'Core.L1_Foundation.Foundation',
    r'\bCore\.L1_Foundation\.Foundation\.Ethics\.Ethics\b': 'Core.L1_Foundation.Foundation.Ethics'
}

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for pattern, replacement in replacements.items():
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"[FIXED] {filepath}")
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")

def main():
    base_dir = r"c:\Elysia"
    for root, dirs, files in os.walk(base_dir):
        if any(d in root for d in [".git", ".venv", "__pycache__"]):
            continue
        for file in files:
            if file.endswith(".py"):
                fix_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
