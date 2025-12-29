import os
import shutil

def migrate_files():
    BASE = r"c:\Elysia\Core\FoundationLayer\Foundation"
    SRC = os.path.join(BASE, "Math")
    DST = os.path.join(BASE, "Wave")
    
    if not os.path.exists(SRC):
        print("❌ Source Math directory not found.")
        return
        
    print(f"📦 Migrating files from {SRC} -> {DST}")
    
    for filename in os.listdir(SRC):
        if filename == "__init__.py" or filename == "__pycache__":
            continue
            
        src_file = os.path.join(SRC, filename)
        dst_file = os.path.join(DST, filename)
        
        if os.path.isfile(src_file):
            if os.path.exists(dst_file):
                print(f"   ⚠️ Skipping {filename} (Exists in Dest)")
            else:
                shutil.move(src_file, dst_file)
                print(f"   ✅ Moved {filename}")
                
    print("🎉 Migration complete.")

if __name__ == "__main__":
    migrate_files()
