import os

def replace_in_files(directory):
    for root, dirs, files in os.walk(directory):
        if '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.md') or file.endswith('.json') or file == '.gitignore':
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if 'elysia_cortex' in content:
                        content = content.replace('elysia_cortex', 'elysia_cortex')
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Updated: {filepath}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    replace_in_files(r'C:\Elysia')
    replace_in_files(r'C:\elysia_seed')
    replace_in_files(r'C:\elysia_cortex')
    print("All string replacements finished.")
