import os

path = r'C:/Elysia\Core\L1_Foundation\Foundation\hyper_cosmos.py'
try:
    with open(path, 'rb') as f:
        content = f.read()
    
    # Attempt decoding
    try:
        text = content.decode('utf-8')
        print("Successfully decoded as UTF-8")
    except UnicodeDecodeError:
        try:
            text = content.decode('cp949')
            print("Successfully decoded as CP949")
        except UnicodeDecodeError:
            text = content.decode('utf-8', errors='replace')
            print("Decoded as UTF-8 with replacement for corrupted characters")
    
    # Save as clean UTF-8
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f"Fixed encoding for {path}")

except Exception as e:
    print(f"Error fixing file: {e}")
