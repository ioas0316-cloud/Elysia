import os
import re

replacements = {
    r'\bCore\.L3_Logos\b': 'Core.Phenomena',
    r'\bCore\.L5_Intelligence\b': 'Core.Cognition',
    r'\bCore\.System\b': 'Core.System',
    r'\bCore\.World\b': 'Core.Cognition',
    r'\bCore\.L3_Phenomena\.Vision\b': 'Core.Phenomena',
    r'\bCore\.L3_Phenomena\.Voice\b': 'Core.Phenomena',
    r'\bCore\.L3_Phenomena\.Prism\b': 'Core.Phenomena',
    r'\bCore\.L5_Mental\.Intelligence\b': 'Core.Cognition',
    r'\bCore\.L5_Mental\.Logic\b': 'Core.Cognition',
    r'\bCore\.L7_Spirit\.Monad\b': 'Core.Divine',
    r'\bCore\.L7_Spirit\.Sovereignty\b': 'Core.Divine',
    r'\bCore\.L1_Foundation\.Foundation\.Foundation\b': 'Core.System',
    r'\bCore\.L1_Foundation\.Foundation\.Ethics\.Ethics\b': 'Core.System'
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
