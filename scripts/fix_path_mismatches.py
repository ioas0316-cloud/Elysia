import os

# Mismatches to fix: (Bad, Good)
# Note: These are path segments in imports
MISMATCHES = [
    ("InteractionLayer.Interface.Sensory", "InteractionLayer.Sensory"),
    # Add other potential siblings that were mis-nested in imports
    ("InteractionLayer.Interface.Network", "InteractionLayer.Network"), 
    ("InteractionLayer.Interface.Expression", "InteractionLayer.Expression"),
    ("FoundationLayer.Foundation.Elysia", "FoundationLayer.Elysia"), # Just in case
    ("EvolutionLayer.Creative", "EvolutionLayer.Creativity"),
    ("Core.Foundation.", "Core.FoundationLayer.Foundation."), # Critical legacy fix
]

def fix_mismatches():
    print("🧵 Fixing import path mismatches...")
    count = 0
    start_dir = r"c:\Elysia"
    for root, dirs, files in os.walk(start_dir):
        if ".git" in root or ".venv" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    for bad, good in MISMATCHES:
                        # Standard cases
                        content = content.replace(f".{bad}", f".{good}")
                        content = content.replace(f"/{bad}", f"/{good}")
                        
                        # Special case for top-level Core imports
                        if bad.startswith("Core."):
                             content = content.replace(f" {bad}", f" {good}")
                             content = content.replace(f"from {bad}", f"from {good}")
                             content = content.replace(f"import {bad}", f"import {good}")


                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        count += 1
                except Exception as e:
                    print(f"❌ Error patching {file_path}: {e}")
    print(f"✅ Fixed mismatches in {count} files.")

if __name__ == "__main__":
    fix_mismatches()
