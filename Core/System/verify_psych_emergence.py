import requests
import json
import time
import collections

SERVER_URL = "http://localhost:8000"

def verify_emergence():
    print(f"🌱 [VERIFY] Connecting to Psych-Field Reality Engine...")
    
    # 1. Wait for Simulation to Settle (Particles finding niches)
    print("   Waiting 5 seconds for psycho-social drift...")
    time.sleep(5)
    
    try:
        res = requests.get(f"{SERVER_URL}/state")
        if res.status_code != 200:
            print(f"❌ Connection Failed: {res.status_code}")
            return
            
        data = res.json()
        entities = data.get("entities", []) # This might be named differently in harvest_snapshot, let's check structure
        
        # 'harvest_snapshot' usually returns a dict with 'entities' or 'cells'
        # In World class it returns { "timestamp":..., "cells": [...] }
        cells = data.get("cells", []) 
        
        if not cells:
            # Fallback for RealityServer's get_snapshot flat list logic?
            # RealityServer currently returns world.harvest_snapshot
            pass
            
        print(f"   Analyzed {len(cells)} Souls.")
        
        # 2. Analyze Role Distribution
        roles = collections.Counter()
        trait_correlations = collections.defaultdict(list)
        
        for c in cells:
            role = c.get("role", "Unawakened")
            roles[role] += 1
            
            traits = c.get("traits", {})
            if role == "Sage":
                trait_correlations["Sage_Mind"].append(traits.get("mind_pref", 0))
            elif role == "Warrior":
                trait_correlations["Warrior_Body"].append(traits.get("body_pref", 0))
                
        # 3. Report Emergence
        print("\n📊 [EMERGENCE REPORT]")
        for role, count in roles.items():
            print(f"   - {role}: {count}")
            
        # 4. Verify Correlations (Did they self-sort correctly?)
        print("\n🔍 [CORRELATION CHECK]")
        if trait_correlations["Sage_Mind"]:
            avg_mind = sum(trait_correlations["Sage_Mind"]) / len(trait_correlations["Sage_Mind"])
            print(f"   - Average 'Mind Preference' of Sages: {avg_mind:.2f} (Should be > 0.0)")
            
        if trait_correlations["Warrior_Body"]:
            avg_body = sum(trait_correlations["Warrior_Body"]) / len(trait_correlations["Warrior_Body"])
            print(f"   - Average 'Body Preference' of Warriors: {avg_body:.2f} (Should be > 0.0)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_emergence()
