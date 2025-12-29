import os
import re

ROOT = r"c:\Elysia\Core"

# Simple replacement map based on clean_data.py logic
# NOTE: This is risky. We need to be precise.
# We will verify file existence before replacing in a real run, but here we just do string replacement.

REPLACEMENTS = [
    # Memory
    ('"data/memory.db"', '"data/Memory/memory.db"'),
    ('"c:\\\\Elysia\\\\data\\\\memory.db"', '"c:\\\\Elysia\\\\data\\\\Memory\\\\memory.db"'),
    ('path="data/memory.db"', 'path="data/Memory/memory.db"'),
    
    ('"data/elysia_core_memory.json"', '"data/Memory/elysia_core_memory.json"'),
    ('"data/conversation_memory.json"', '"data/Memory/conversation_memory.json"'),
    ('"data/synaptic_memory.json"', '"data/Memory/synaptic_memory.json"'),

    # State
    ('"data/brain_state.pt"', '"data/State/brain_state.pt"'),
    ('"c:\\\\Elysia\\\\data\\\\brain_state.pt"', '"c:\\\\Elysia\\\\data\\\\State\\\\brain_state.pt"'),
    ('"data/emergent_self.json"', '"data/State/emergent_self.json"'),
    ('"c:\\\\Elysia\\\\data\\\\emergent_self.json"', '"c:\\\\Elysia\\\\data\\\\State\\\\emergent_self.json"'),
    
    # Logs
    ('"data/growth_history.json"', '"data/Logs/growth_history.json"'),
    ('"data/awakening_log.txt"', '"data/Logs/awakening_log.txt"'),
    
    # Knowledge
    ('"data/hierarchical_knowledge.json"', '"data/Knowledge/hierarchical_knowledge.json"'),
    ('"data/potential_knowledge.json"', '"data/Knowledge/potential_knowledge.json"'),
    ('"data/concept_dictionary.json"', '"data/Knowledge/concept_dictionary.json"'),
    ('"data/unified_spatial_index.json"', '"data/Knowledge/unified_spatial_index.json"'),
    ('"data/tools_kg.json"', '"data/Knowledge/tools_kg.json"'),
]

def fix_data_paths():
    print("🔧 Fixing hardcoded data paths...")
    count = 0
    
    for root, dirs, files in os.walk(ROOT):
        for file in files:
            if file.endswith(".py") or file.endswith(".md"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    for old, new in REPLACEMENTS:
                        new_content = new_content.replace(old, new)
                        
                    if new_content != content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                        print(f"   Fixed: {os.path.basename(path)}")
                except Exception as e:
                    pass
                    
    print(f"✅ Updated paths in {count} files.")

if __name__ == "__main__":
    fix_data_paths()
