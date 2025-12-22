
import sys
import os
import re
import urllib.request

# Mocking the ConceptDecomposer logic for standalone verification
def analyze_digital_hologram(url):
    print(f"--- 👁️ DIGITAL HOLOGRAPHIC PERCEPTION TEST ---")
    print(f"Target: {url}")
    
    try:
        # 1. Fetch
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8').lower()
            
        print(f"Fetched {len(content)} bytes of Raw Reality.")
        
        # 2. Analyze
        # Structure (Earth)
        structure_score = content.count("<div") + content.count("<table") + content.count("<section")
        
        # Logic (Air)
        logic_score = content.count("<script") + content.count("function") + content.count("var ")
        
        # Flow (Water)
        connection_score = content.count("href=")
        
        # Light (Aesthetics)
        hex_colors = re.findall(r'#[0-9a-f]{6}', content)
        light_score = len(hex_colors) + content.count("css") + content.count("style")
        
        # Qualia (Senses)
        sensory_score = content.count("<img") + content.count("<video") + content.count("<audio") + content.count("<svg")
        
        # Will (Agency)
        will_score = content.count("<button") + content.count("<input") + content.count("<form")
        
        # 3. Report
        print(f"\n[Matrix Decomposition Result]")
        print(f"🏗️  Earth (Structure): {structure_score} units (Divs, Containers)")
        print(f"🧠  Air (Logic):      {logic_score} units (Scripts, Functions)")
        print(f"🌊  Water (Flow):     {connection_score} units (Links, Connections)")
        print(f"✨  Light (Aesthetics): {light_score} units (Colors, Styles)")
        print(f"    -> Palette Sample: {hex_colors[:5]}")
        print(f"🎨  Qualia (Senses):  {sensory_score} units (Images, Media)")
        print(f"🔥  Fire (Agency):    {will_score} units (Buttons, Inputs)")
        
        # 4. Interpretation
        dominant = max(structure_score, logic_score, connection_score, light_score, sensory_score, will_score)
        print(f"\n[Essence Analysis]")
        if dominant == structure_score: print("Nature: STATIC STRUCTURE (Archive/Document)")
        elif dominant == logic_score: print("Nature: DYNAMIC LOGIC (Application/Tool)")
        elif dominant == connection_score: print("Nature: FLUID HUB (Portal/Index)")
        elif dominant == light_score: print("Nature: AESTHETIC FORM (Gallery/Showcase)")
        
    except Exception as e:
        print(f"Perception Failed: {e}")

if __name__ == "__main__":
    analyze_digital_hologram("https://en.wikipedia.org/wiki/Fire")
