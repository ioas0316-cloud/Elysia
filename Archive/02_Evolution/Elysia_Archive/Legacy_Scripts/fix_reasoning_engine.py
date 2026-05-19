import os

def fix_reasoning():
    print("🧠 Fixing ReasoningEngine imports...")
    count = 0
    start_dir = r"c:\Elysia"
    # Bad pattern
    bad = "Core.Cognition.Reasoning.reasoning_engine"
    # Good pattern
    good = "Core.Cognition.Reasoning.reasoning_engine"
    
    for root, dirs, files in os.walk(start_dir):
        if ".git" in root or ".venv" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if bad in content:
                        new_content = content.replace(bad, good)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                        print(f"   Patched: {file}")
                        
                except Exception as e:
                    print(f"❌ Error patching {file_path}: {e}")
                    
    print(f"✅ Fixed {count} files.")

if __name__ == "__main__":
    fix_reasoning()
