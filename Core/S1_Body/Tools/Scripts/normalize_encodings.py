
import os

def normalize_file(path):
    encodings = ['utf-8', 'cp949', 'latin-1', 'utf-16']
    content = None
    used_enc = None
    
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
                used_enc = enc
                break
        except:
            continue
            
    if content is None:
        return

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
         print(f"Failed to write {path}: {e}")

for root, dirs, files in os.walk('c:/Elysia/Core'):
    for file in files:
        if file.endswith('.py'):
            normalize_file(os.path.join(root, file))
