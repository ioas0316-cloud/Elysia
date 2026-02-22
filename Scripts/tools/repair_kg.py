import os
import json

def repair_kg(path):
    print(f"🔧 [REPAIR] Inspecting {path}...")
    try:
        with open(path, 'rb') as f:
            content = f.read()
        
        # Look for the last valid node boundary '}},'
        last_node = content.rfind(b'}},')
        if last_node != -1:
            print(f"✅ Found node boundary at {last_node}. Truncating and closing...")
            repaired = content[:last_node+2] + b'}, "edges": []}'
            with open(path, 'wb') as f:
                f.write(repaired)
            
            # Verify
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
            print("✨ [REPAIR] Success. Knowledge Graph is now valid.")
            return True
        else:
            print("❌ Could not find node boundary '}},'. Trying fallback...")
            last_brace = content.rfind(b'}')
            if last_brace != -1:
                repaired = content[:last_brace+1]
                with open(path, 'wb') as f:
                    f.write(repaired)
                print("⚠️ [REPAIR] Forced closure applied. Integrity uncertain.")
                return True
    except Exception as e:
        print(f"💥 [REPAIR] Failed: {e}")
    return False

if __name__ == "__main__":
    repair_kg('data/kg_with_embeddings.json')
