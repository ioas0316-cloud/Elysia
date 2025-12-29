import os

REDUNDANCIES = [
    ("Expression.Expression", "Expression"),
    ("Network.Network", "Network"),
    ("Intelligence.Intelligence", "Intelligence"),
    ("Interaction.Interaction", "Interaction"),
    ("Foundation.Foundation", "Foundation"),
    ("Interface.Interface", "Interface"),
    ("Creativity.Creativity", "Creativity"),
]

def fix_redundancies():
    print("🧵 Fixing redundant imports...")
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
                    
                    for bad, good in REDUNDANCIES:
                        # Replace bad with good
                        # Be careful about context, but usually dot notation is safe here
                        content = content.replace(f".{bad}", f".{good}")
                        # Also slash notation just in case
                        content = content.replace(f"/{bad}", f"/{good}")

                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        count += 1
                except Exception as e:
                    print(f"❌ Error patching {file_path}: {e}")
    print(f"✅ Fixed duplications in {count} files.")

if __name__ == "__main__":
    fix_redundancies()
