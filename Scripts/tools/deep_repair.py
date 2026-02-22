import os
import json

def deep_repair(path):
    print(f"🧹 [DEEP_REPAIR] Auditing {path}...")
    try:
        with open(path, 'rb') as f:
            data = f.read()
        
        # Look for the last complete node entry boundary.
        # Nodes look like: \"id\": { ... }}
        # So we look for }} followed by a comma, or just }}
        target = b'}}'
        idx = data.rfind(target)
        
        if idx != -1:
            # We truncate after the last complete }}
            # Then we need to see where we are. If we are in the 'nodes' dict:
            # { \"nodes\": { ... }} <- we are here.
            # We need to add a closing brace for the 'nodes' dict, then the root dict.
            # But wait, let's just close it as a valid dict.
            repaired = data[:idx+2]
            
            # Count opening vs closing braces to be sure
            open_braces = repaired.count(b'{')
            close_braces = repaired.count(b'}')
            
            diff = open_braces - close_braces
            print(f"📊 [DEEP_REPAIR] Brace Diff: {diff}")
            
            for _ in range(diff):
                repaired += b'}'
            
            with open(path, 'wb') as f:
                f.write(repaired)
            
            # Final Verification
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
            print("✨ [DEEP_REPAIR] Success. Manifold is stable.")
            return True
        else:
            print("❌ No valid boundaries found.")
    except Exception as e:
        print(f"💥 [DEEP_REPAIR] Error during verification: {e}")
        # If verify failed, it's likely we need one less or one more brace.
        # Let's try to just find the LAST valid JSON object from the start.
    return False

if __name__ == "__main__":
    deep_repair('data/kg_with_embeddings.json')
