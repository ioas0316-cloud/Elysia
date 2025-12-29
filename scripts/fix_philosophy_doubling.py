import os

def fix_philosophy():
    print("📜 Fixing Philosophy doublings...")
    count = 0
    start_dir = r"c:\Elysia"
    # Bad pattern
    bad = "Core.FoundationLayer.Philosophy"
    # Good pattern
    good = "Core.FoundationLayer.Philosophy"
    
    # Also fix aesthetic_principles -> aesthetic_principles
    bad_file = "aesthetic_principles"
    good_file = "aesthetic_principles"

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
                    
                    # Fix doubling
                    if bad in content:
                        content = content.replace(bad, good)
                        
                    # Fix file name
                    if bad_file in content:
                        content = content.replace(bad_file, good_file)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        count += 1
                        print(f"   Patched: {file}")
                        
                except Exception as e:
                    print(f"❌ Error patching {file_path}: {e}")
                    
    print(f"✅ Fixed {count} files.")

if __name__ == "__main__":
    fix_philosophy()
