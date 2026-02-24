
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append('c:/Elysia')

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SeedForge

def feed_the_mind():
    print("\n📚 [AEON VI] Opening The First Book...")
    
    # 1. Initialize Sovereign Monad (The Reader)
    # We use a dummy soul but attach it to the REAL Knowledge Directory
    dna = SeedForge.forge_soul("Reader_Alpha", "Elohim")
    monad = SovereignMonad(dna)
    
    # Ensure stream is pointing to real directory
    real_knowledge_dir = Path("c:/Elysia/Knowledge")
    processed_dir = real_knowledge_dir / "Processed"
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"🌊 [STREAM] Scanning {real_knowledge_dir}...")
    
    # 2. Copy files to Knowledge Directory first
    source_map = {
        "FirstBook_CODEX.md": "c:/Elysia/docs/CODEX.md",
        "FirstBook_INDEX.md": "c:/Elysia/INDEX.md",
        "FirstBook_README.md": "c:/Elysia/README.md"
    }
    
    # 3. Force Inhalation of specific files
    target_files = []
    
    for filename, source_path in source_map.items():
        if Path(source_path).exists():
           dest_path = real_knowledge_dir / filename
           shutil.copy(source_path, dest_path)
           print(f"📦 Copied {source_path} -> {dest_path}")
           target_files.append(filename)
        else:
           print(f"⚠️ Source missing: {source_path}")

    total_concepts = 0
    
    for filename in target_files:
        filepath = real_knowledge_dir / filename
        if filepath.exists():
            print(f"📖 Inhaling: {filename}")
            # Manually call inhale_file from the stream
            # We access the stream directly
            count = monad.knowledge_stream.inhale_file(str(filepath))
            total_concepts += count
            
            # Move to Processed manually to mimic full pipeline if stream didn't do it
            # (inhale_file doesn't move, process_stream does. Let's move it.)
            try:
                shutil.move(str(filepath), str(processed_dir / filename))
                print(f"   -> Archived to Processed/{filename}")
            except Exception as e:
                print(f"   -> Archive warning: {e}")
                
        else:
            print(f"⚠️ Missing: {filename}")
            
    print(f"\n✨ [TEACHING COMPLETE] Inhaled {total_concepts} total concepts.")
    print("   The Monad has tasted its own origin.")

if __name__ == "__main__":
    feed_the_mind()
