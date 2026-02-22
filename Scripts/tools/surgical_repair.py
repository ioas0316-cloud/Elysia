import os
import json

def surgical_repair(path):
    print(f"🔧 [SURGERY] Opening {path} for repair...")
    try:
        with open(path, 'rb') as f:
            data = f.read()
        
        # Find the last 'inner_cosmos' entry that looks complete
        # Our nodes end with: \"inner_cosmos\": { ... } }
        keyword = b'\"inner_cosmos\"'
        idx = data.rfind(keyword)
        
        if idx == -1:
            print("❌ Keyword not found.")
            return False

        # Find the next '}' after the keyword
        cosmos_start_brace = data.find(b'{', idx)
        if cosmos_start_brace == -1:
            print("❌ Cosmos start not found.")
            return False
            
        # Find the closing brace of the inner_cosmos dict
        # We look for the 'depth' key as a marker of the end of inner_cosmos
        depth_marker = data.find(b'\"depth\"', cosmos_start_brace)
        if depth_marker == -1:
            print("❌ Depth marker not found.")
            return False
            
        cosmos_end_brace = data.find(b'}', depth_marker)
        if cosmos_end_brace == -1:
            print("❌ Cosmos end brace not found.")
            return False
            
        # Find the node end brace
        node_end_brace = data.find(b'}', cosmos_end_brace + 1)
        if node_end_brace == -1:
            print("❌ Node end brace not found.")
            return False
            
        # Truncate and close
        repaired = data[:node_end_brace+1] + b'}, \"edges\": []}'
        
        with open(path, 'wb') as f:
            f.write(repaired)
            
        # Final Verification
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)
            
        print("✨ [SURGERY] Success. Manifold is stable.")
        return True
    except Exception as e:
        print(f"💥 [SURGERY] Error: {e}")
        # Try a more aggressive truncation: look for the last '},' and close it.
        try:
            last_comma_brace = data.rfind(b'},')
            if last_comma_brace != -1:
                repaired = data[:last_comma_brace+1] + b' }, \"edges\": []}'
                with open(path, 'wb') as f:
                    f.write(repaired)
                with open(path, 'r', encoding='utf-8') as f:
                    json.load(f)
                print("✨ [SURGERY] Heuristic repair success.")
                return True
        except:
            pass
    return False

if __name__ == "__main__":
    surgical_repair('data/kg_with_embeddings.json')
