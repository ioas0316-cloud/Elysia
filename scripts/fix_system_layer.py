import os

def fix_system():
    print("🔧 Fixing SystemLayer imports...")
    count = 0
    start_dir = r"c:\Elysia"
    # Bad pattern
    bad = "Core.SystemLayer.System"
    # Good pattern
    good = "Core.SystemLayer.System"
    
    # Also handle the broader Monitoring segment if needed, but be specific first
    bad2 = "Core.SystemLayer.Monitoring"
    good2 = "Core.SystemLayer.System" # This might be risky if Structure is different

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
                    
                    if bad in content:
                        content = content.replace(bad, good)
                    
                    # Fallback for just Monitoring if it points to System
                    # Note: Check if Monitoring is used for other things
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        count += 1
                        print(f"   Patched: {file}")
                        
                except Exception as e:
                    print(f"❌ Error patching {file_path}: {e}")
                    
    print(f"✅ Fixed {count} files.")

if __name__ == "__main__":
    fix_system()
