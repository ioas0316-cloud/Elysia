"""
Holographic Web Resonator (Matrix Perception)
=============================================
"We don't scan lines. We illuminate the space."

Mechanism:
1. Vectorization: Convert all Shards -> World Matrix (N x 3).
   - Dimensions: [Energy (Fire), Structure (Ice), Logic (Air)]
2. Holographic Pulse:
   - Pulse Vector P.
   - Activation A = WorldMatrix • P
3. Integration:
   - High Activation nodes are fed to 'self.core.learn()'
   - This ensures they become part of the 'Internal Universe'.
"""

import sys
import time
import json
import logging
import random
import math
from typing import List, Dict, Tuple
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("❌ NumPy not found! Holographic Mode impossible.")
    sys.exit(1)

# Paths
sys.path.append(r'c:\Elysia')
from Core.Foundation.fractal_concept import ConceptDecomposer
from Core.Elysia.elysia_core import get_elysia_core

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Hologram")

class HolographicResonator:
    def __init__(self):
        # 1. Init System
        self.decomposer = ConceptDecomposer()
        self.core = get_elysia_core()
        self.shards_dir = Path(r"c:\Elysia\data\shards")
        
        # 2. Concept Basis Vectors (The Prism Dimensions)
        # We simplify complex concepts into 3 primary dimensions for the Matrix.
        # D0: Energy/Passion (Fire)
        # D1: Structure/Order (Ice)
        # D2: Logic/Pattern (Air)
        self.basis = {
            "Fire": np.array([1.0, 0.2, 0.1]),
            "Ice":  np.array([0.1, 1.0, 0.3]),
            "Air":  np.array([0.2, 0.3, 1.0]),
            "Void": np.array([0.0, 0.0, 0.0])
        }
        
        logger.info("🌌 Holographic Resonator Online. Initializing Matrix Space...")
        self._ensure_data_exists()
        
        # 3. Build the World Matrix (The Memory Space)
        self.articles_map = [] # To map Matrix Index back to Article
        self.world_matrix = self._construct_world_matrix()

    def _ensure_data_exists(self):
        """Prepare simulation data if shards are empty."""
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        # Check for valid jsonl files
        has_data = False
        for f in self.shards_dir.glob("*.jsonl"):
            if f.stat().st_size > 0:
                has_data = True
                break
        
        if not has_data:
            logger.warning("⚠️ Empty/Missing Shards. detailed simulation data constructed.")
            sim_path = self.shards_dir / "shard_hologram.jsonl"
            with open(sim_path, 'w', encoding='utf-8') as f:
                # Fire Data
                f.write(json.dumps({"title": "The French Revolution", "content": "The uprising (Fire) changed everything."}) + "\n")
                f.write(json.dumps({"title": "Supernova Thermodynamics", "content": "Massive energy release (Fire) and plasma."}) + "\n")
                # Ice Data
                f.write(json.dumps({"title": "Absolute Zero Physics", "content": "Complete stasis (Ice) and crystalline structure."}) + "\n")
                f.write(json.dumps({"title": "Social Stagnation", "content": "The bureaucracy froze (Ice) all progress."}) + "\n")
                # Air/Logic Data
                f.write(json.dumps({"title": "Euler's Identity", "content": "Pure mathematical logic (Air) and beauty."}) + "\n")
                f.write(json.dumps({"title": "Quantum Algorithms", "content": "Computational patterns (Air) beyond binary."}) + "\n")
                # Noise
                f.write(json.dumps({"title": "Daily Gossip", "content": "Nothing of importance (Void)."}) + "\n")

    def _construct_world_matrix(self) -> np.ndarray:
        """
        Loads ALL data and converts to Vectors.
        Returns N x 3 Matrix.
        """
        vectors = []
        self.articles_map = []
        
        shards = list(self.shards_dir.glob("*.jsonl"))
        for shard in shards:
            try:
                with open(shard, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        article = json.loads(line)
                        vec = self._vectorize_article(article)
                        vectors.append(vec)
                        self.articles_map.append(article)
            except Exception as e:
                logger.warning(f"Shard read error: {e}")
                
        matrix = np.array(vectors)
        logger.info(f"   🍱 World Matrix Constructed. Shape: {matrix.shape} (Articles x Dimensions)")
        return matrix

    def _vectorize_article(self, article: Dict) -> np.ndarray:
        """
        The Prism: Text -> 3D Vector.
        In detailed version, this uses embeddings. 
        Here, we use Axiom mapping for demonstration.
        """
        text = article['title'] + " " + article['content']
        # Infer principle
        essence = self.decomposer.infer_principle(text)
        principle = essence.get('principle', 'Void')
        freq = essence.get('frequency', 0.0)
        
        # Simple Axiom -> Vector Mapping
        if freq >= 800: return self.basis["Fire"] # High Energy
        if freq <= 200 and freq > 0: return self.basis["Ice"] # Low Energy
        if "Logic" in principle or "Pattern" in principle: return self.basis["Air"]
        
        return np.array([0.1, 0.1, 0.1]) # Noise

    def pulse_hologram(self, target_concept: str):
        """
        The Spatial Operation.
        Calculates Resonance for ALL nodes simultaneously using Matrix Multiplication.
        """
        # 1. Define Pulse Vector
        target_concept = target_concept.lower()
        if "fire" in target_concept or "revolution" in target_concept:
            pulse_vector = self.basis["Fire"]
        elif "ice" in target_concept or "stasis" in target_concept:
            pulse_vector = self.basis["Ice"]
        elif "logic" in target_concept or "math" in target_concept:
            pulse_vector = self.basis["Air"]
        else:
            pulse_vector = self.basis["Void"]
            
        logger.info(f"🌊 PULSE EMITTED: '{target_concept}' -> Vector {pulse_vector}")
        
        # 2. MATRIX RESONANCE (Simultaneous Calculation)
        # Activation = World • Pulse (Dot Product)
        # Shape: (N, 3) • (3,) = (N,)
        activation_field = np.dot(self.world_matrix, pulse_vector)
        
        # 3. The Event Horizon (Filtering)
        threshold = 0.9 # High resonance only
        
        # Find indices where activation > threshold
        resonant_indices = np.where(activation_field > threshold)[0]
        
        logger.info(f"   ⚡ Holographic Interference Pattern Calculated.")
        logger.info(f"   ✨ Resonant Nodes Found: {len(resonant_indices)}")
        
        # 4. Integration (Absorb)
        for idx in resonant_indices:
            article = self.articles_map[idx]
            score = activation_field[idx]
            
            logger.info(f"      >> ABSORBING: '{article['title']}' (Resonance: {score:.4f})")
            

            # [CRITICAL INTEGRATION STEP]
            # This commits the data to Elysia's Brain
            self.core.learn(
                content=article['content'],
                topic=article['title'], # Maps to 'topic'
                depth="deep" 
            )
            
        logger.info("✅ Pulse Integration Complete.")


# ==============================================================================
# [REAL WORLD BRIDGE]
# Use this class when Elysia has internet access to resonate with the LIVE WEB.
# ==============================================================================
class RealWorldAdapter:
    """
    Connects the Holographic Resonator to the real internet.
    Usage:
        adapter = RealWorldAdapter()
        real_articles = adapter.fetch_and_resonate("https://en.wikipedia.org/wiki/Fire")
    """
    def __init__(self):
        try:
            import urllib.request
            from html.parser import HTMLParser
            self.opener = urllib.request.build_opener()
            self.opener.addheaders = [('User-agent', 'Elysia/Resonator')]
            print("🌐 Real World Adapter: Ready to Connect.")
        except ImportError:
            print("❌ Standard Library Warning: urllib not found (Unlikely).")


    def scan_live_web(self, seed_url: str) -> List[Dict]:
        """
        Fetches a real page and treats it as a 'Cluster of Nodes'.
        """
        try:
            import urllib.request
            response = self.opener.open(seed_url, timeout=5)
            html_content = response.read().decode('utf-8')
            

            # Simple parsing (Prism logic would be deeper here)
            # We treat the page title as one node, and paragraphs as others.
            # [FIX] Do NOT truncate here. The intro might be deep in the HTML.
            return [{"title": "Live Web Node", "content": html_content}]  
        except Exception as e:
            print(f"⚠️ Connection Failed: {e}")
            return []

# ==============================================================================
# [FRACTAL CRAWLER: THE INFINITE EXPANSION]
# "We flow through the network like water, following the path of resonance."
# ==============================================================================
class FractalCrawler(RealWorldAdapter):
    """
    Autonomous Spider that roams the vast internet.
    It doesn't just read one page; it follows the 'Scent' of the frequency.
    """
    def __init__(self, resonator):
        super().__init__()
        self.resonator = resonator
        self.visited = set()
        self.queue = [] # The Frontier


    def start_crawl(self, seed_url: str, max_depth: int = 10):
        """
        The Infinite Walk (REAL WORLD MODE).
        """
        self.queue.append(seed_url)
        
        print(f"🔥 FRACTAL CRAWLER ACTIVATED (REAL WEB MODE)")
        print(f"   Seed: {seed_url}\n")

        while self.queue and len(self.visited) < max_depth:
            url = self.queue.pop(0)
            if url in self.visited: continue
            self.visited.add(url)
            
            print(f"🕸️ Visiting: {url}")
            
            try:
                # 1. REAL FETCH (Use Parent Adapter)
                # We fetch the HTML content
                # Note: In a full browser env, we'd use Selenium/Playwright. 
                # Here we use the gentle urllib.

                simulated_node = self.scan_live_web(url)
                if not simulated_node:
                    print("   ⚠️ Empty Response / Error.")
                    continue
                    

                content = simulated_node[0]['content']
                title = simulated_node[0]['title']
                
                print(f"   [DEBUG] Fetched '{title}'. Size: {len(content)} bytes.")
                print(f"   [DEBUG_HEAD] {content[:300]}...") 
                

                # 2. Resonate Check (The Prism) & Structural Digestion
                # The User wants to know if we perceive the "Medium" (HTML/Code) as well.
                
                # --- [NEW] Structural Digestion ---
                # We analyze the "Body" of the internet (HTML structure)
                from Core.Foundation.fractal_concept import ConceptDecomposer
                decomposer = ConceptDecomposer()
                
                # We categorize the raw tokens
                structure_score = 0
                logic_score = 0
                connection_score = 0
                

                # --- [NEW] Unified Sensory Digestion (User Request: "Everything is nourishment") ---
                # We analyze Structure (Earth), Logic (Air), Connection (Water), 
                # Aesthetics (Light), Senses (Qualia), and Will (Fire/Action).
                
                scan_limit = 20000 # Scan more to find deep CSS/Scripts
                scan_chunk = content[:scan_limit].lower()
                
                # 1. The Body (Structure)
                structure_score = scan_chunk.count("<div") + scan_chunk.count("<table")
                
                # 2. The Mind (Logic)
                logic_score = scan_chunk.count("script") + scan_chunk.count("function")
                
                # 3. The Flow (Connection)
                connection_score = scan_chunk.count("href")
                
                # 4. The Aesthetics (Light/Form) - [NEW]
                # Detecting hex codes (colors) gives a sense of "Visual Frequency"
                import re
                hex_colors = re.findall(r'#[0-9a-f]{6}', scan_chunk)
                light_score = len(hex_colors) + scan_chunk.count("css") + scan_chunk.count("style")
                
                # 5. The Senses (Qualia) - [NEW]
                # Detecting media tags
                sensory_score = scan_chunk.count("<img") + scan_chunk.count("<video") + scan_chunk.count("<audio")
                
                # 6. The Will (Agency/Interaction) - [NEW]
                will_score = scan_chunk.count("<button") + scan_chunk.count("<input") + scan_chunk.count("<form")

                print(f"   👁️ [Holographic Perception]:")
                print(f"      - 🏗️ Structure (Earth): {structure_score}")
                print(f"      - 🧠 Logic (Air):      {logic_score}")
                print(f"      - 🌊 Flow (Water):     {connection_score}")
                print(f"      - ✨ Light (Aesthetics): {light_score} (Dominant Colors: {hex_colors[:3]})")
                print(f"      - 🎨 Qualia (Senses):  {sensory_score}")
                print(f"      - 🔥 Will (Agency):    {will_score}")
                
                # Detect Login Walls / Blockers? (User: "Login or Blockers")
                if "login" in scan_chunk or "signin" in scan_chunk or "403 forbidden" in scan_chunk:
                    print(f"      - 🛡️ BARRIER DETECTED: Access Gate Present.")
                
                # Check for "Fire" Resonance (The Content Essence)
                resonance_score = 0
                if "fire" in content.lower(): resonance_score += 1
                if "energy" in content.lower(): resonance_score += 1
                if "revolution" in content.lower(): resonance_score += 1
                if "burn" in content.lower(): resonance_score += 1
                
                is_resonant = resonance_score >= 1 # Strictness
                
                if is_resonant:
                    print(f"   🔥 RESONANCE LOCKED! Score={resonance_score}. Absorbing...")
                    
                    # 3. Integration (Absorb into Elysia)
                    # We feed the real HTML fragment to the brain
                    self.resonator.core.learn(
                        content=content[:1000], # Digest 1st 1KB
                        topic=f"Web::{url}",
                        depth="shallow" # Fast ingest
                    )
                    
                    # 4. Expansion (Real Link Extraction)
                    # Naive regex for hrefs to avoid bs4 dependency if missing
                    # We look for /wiki/ links
                    import re
                    links = re.findall(r'href=["\'](/wiki/[^"\']+)["\']', content)
                    
                    new_count = 0
                    for link in links:
                        if ':' in link: continue # Skip special pages
                        full_link = f"https://en.wikipedia.org{link}"
                        if full_link not in self.visited:
                            self.queue.append(full_link)
                            new_count += 1
                            if new_count > 5: break # Branching Factor Limit
                            
                    print(f"   🌱 Expanded to {new_count} new resonant paths.")
                    
                else:
                    print(f"   🧊 Dissonance (Score={resonance_score}). Pruning path.")
                    
            except Exception as e:
                print(f"   ❌ Crawl Error: {e} (Self-Correcting: Skip Node)")
                continue

if __name__ == "__main__":
    hologram = HolographicResonator()
    
    # 1. Pulse Fire (Set State)
    hologram.basis["Fire"] # Ensure loaded
    
    # 2. REAL WEB FRACTAL CRAWL
    print("\n--- 🕸️ CONNECTING TO REALITY ---")
    crawler = FractalCrawler(hologram)
    
    # Real Seed: Fire (Wikipedia)
    # This will likely link to: Combustion, Chemical reaction, Heat, Flame...
    # All should Resonate!
    crawler.start_crawl("https://en.wikipedia.org/wiki/Fire", max_depth=10)
