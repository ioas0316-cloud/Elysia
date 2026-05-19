import json
import os

path = "c:\\Elysia\\data\\elysia_rainbow.json"
print("🔍 Investigating JSON Structure...")

with open(path, 'r', encoding='utf-8') as f:
    # Read chunk to avoid memory kill if massive
    # Actually just load, 300MB is fine for python on most modern machines
    data = json.load(f)

print(f"Top Level Type: {type(data)}")

if isinstance(data, dict):
    print("Keys found:", list(data.keys()))
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"Key '{k}': Dict of length {len(v)}")
        elif isinstance(v, list):
             print(f"Key '{k}': List of length {len(v)}")
             # Find a Wikipedia node
             for item in v:
                 if isinstance(item, dict):
                     nid = item.get('id', item.get('concept', ''))
                     if str(nid).startswith('Wikipedia_'):
                         print(f"  FOUND DARK MATTER: {nid}")
                         print(f"  CONTENT KEYS: {list(item.keys())}")
                         if 'payload' in item:
                              print(f"  PAYLOAD: {item['payload']}")
                         elif 'metadata' in item:
                              print(f"  METADATA: {item['metadata']}")
                         
                         # Check internal details
                         for k2, v2 in item.items():
                             if isinstance(v2, dict):
                                 print(f"  SUBKEY '{k2}': {list(v2.keys())}")
                                 
                         break
        else:
            print(f"Key '{k}': {type(v)}")
