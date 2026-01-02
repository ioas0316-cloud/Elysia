import os

def fix_imports():
    root_dir = r"c:\Elysia"
    old_import = "from Core.Foundation.Wave.resonance_field import ResonanceField"
    new_import = "from Core.Foundation.Wave.resonance_field import ResonanceField"
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if old_import in content:
                        new_content = content.replace(old_import, new_import)
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"✅ Fixed: {path}")
                        count += 1
                except Exception as e:
                    print(f"⚠️ Error processing {path}: {e}")
    
    print(f"\nTotal files fixed: {count}")

if __name__ == "__main__":
    fix_imports()
