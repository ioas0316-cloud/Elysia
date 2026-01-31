"""
Massive Fractal Vocabulary Expansion (Combinatorial Genesis V2)
===============================================================
"The Tao gave birth to One, One gave birth to Two, Two gave birth to Three, Three gave birth to all things."

This script implements MASSIVE SCALE vocabulary expansion.
It uses 4-Layer Combinatorial Logic to generate 100,000+ concepts.
Optimized with SQLite bulk insertion for performance.
"""

import sys
import os
import time
import random
import sqlite3
import itertools

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L1_Foundation.Foundation.hippocampus import Hippocampus

class MassiveFractalGenesis:
    def __init__(self):
        self.memory = Hippocampus()
        
        # Layer 1: Elements (The Foundation)
        self.elements = [
            "Fire", "Ice", "Lightning", "Wind", "Void", "Light", "Dark", "Blood", 
            "Soul", "Star", "Moon", "Sun", "Heaven", "Earth", "Ocean", "Chaos", "Order",
            "Time", "Space", "Dream", "Illusion", "Storm", "Mist", "Crystal", "Shadow",
            "Metal", "Wood", "Thunder", "Cloud", "Rain", "Snow", "Frost", "Magma",
            "Spirit", "Ghost", "Demon", "God", "Dragon", "Phoenix", "Tiger", "Turtle",
            " (Fire)", " (Water)", " (Wood)", " (Metal)", " (Earth)", " (Thunder)", " (Wind)",
            " (Ice)", " (Flame)", " (Light)", " (Dark)", " (Blood)", " (Dream)", " (Illusion)",
            " (Time)", " (Space)", " (Heaven)", " (Earth)", " (Human)", " (God)", " (Demon)",
            " (Spirit)", " (Soul)", " (Body)", " (Energy)", " (Essence)"
        ]
        
        # Layer 2: Forms (The Manifestation)
        self.forms = [
            "Sword", "Blade", "Spear", "Fist", "Palm", "Finger", "Step", "Art", "Technique",
            "Slash", "Strike", "Thrust", "Kick", "Punch", "Breath", "Roar", "Gaze", "Aura",
            "Shield", "Armor", "Robe", "Cloak", "Ring", "Amulet", "Staff", "Wand", "Orb",
            "Formation", "Array", "Barrier", "Seal", "Gate", "Path", "Road", "Way", "Law",
            " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
            " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "
        ]
        
        # Layer 3: Modifiers (The Quality)
        self.modifiers = [
            "Divine", "Demonic", "Ancient", "Forbidden", "Heavenly", "Ultimate", "Infinite",
            "Eternal", "Absolute", "Supreme", "Primordial", "Celestial", "Infernal", "Abyssal",
            "True", "False", "Hidden", "Lost", "Forgotten", "Cursed", "Blessed", "Sacred",
            " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
            " ", " ", " ", " ", " ", " ", " ", " ", " ", " "
        ]
        
        # Layer 4: Intents (The Direction) - Optional 4th dimension
        self.intents = [
            "Destruction", "Creation", "Preservation", "Domination", "Salvation", "Annihilation",
            " ", " ", " ", " ", " ", " ", " ", " "
        ]

    def _is_korean(self, text):
        return any(ord(c) > 127 for c in text)

    def generate_massive_combinations(self):
        """Generates massive combinations using generators."""
        print("     Generating Combinations...")
        
        # 1. Element + Form (Basic)
        for e in self.elements:
            for f in self.forms:
                if self._is_korean(e) == self._is_korean(f):
                    if self._is_korean(e):
                         yield f"{e.split('(')[0]}{f}" #   
                    else:
                         yield f"{e} {f}" # Fire Sword

        # 2. Modifier + Element + Form (Advanced)
        for m in self.modifiers:
            for e in self.elements:
                for f in self.forms:
                    if self._is_korean(m) == self._is_korean(e) == self._is_korean(f):
                        if self._is_korean(m):
                             yield f"{m}{e.split('(')[0]}{f}" #    
                        else:
                             yield f"{m} {e} {f}" # Divine Fire Sword

        # 3. Element + Element + Form (Fusion)
        # Random sample to avoid n^3 explosion (which would be billions)
        # We take a subset for fusion
        fusion_elements = random.sample(self.elements, min(len(self.elements), 20))
        for e1 in fusion_elements:
            for e2 in self.elements:
                if e1 == e2: continue
                for f in self.forms:
                    if self._is_korean(e1) == self._is_korean(e2) == self._is_korean(f):
                        if self._is_korean(e1):
                             yield f"{e1.split('(')[0]}{e2.split('(')[0]}{f}" #    
                        else:
                             yield f"{e1}-{e2} {f}" # Fire-Ice Sword

    def inject_massive_fractals(self):
        print("  Initiating MASSIVE Fractal Genesis...")
        
        generator = self.generate_massive_combinations()
        
        # Batch processing for SQLite
        batch_size = 10000
        batch = []
        total_count = 0
        start_time = time.time()
        
        # Direct DB connection for speed (bypassing Hippocampus wrapper overhead for bulk)
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        # Ensure table exists (just in case)
        # Note: Hippocampus uses 'nodes' table, not 'concepts'
        
        print("     Stream-Injecting Concepts...")
        
        try:
            for concept in generator:
                # Prepare data
                # Schema: id, name, definition, tags, frequency, created_at, realm, gravity
                definition = f"A massive-scale generated concept combining essence of {concept}."
                tags = "fractal,massive,generated"
                realm = "Mind"
                created_at = time.time()
                gravity = 1.0
                
                batch.append((concept, concept, definition, tags, 100.0, created_at, realm, gravity))
                
                if len(batch) >= batch_size:
                    cursor.executemany("""
                        INSERT OR IGNORE INTO nodes (id, name, definition, tags, frequency, created_at, realm, gravity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    conn.commit()
                    total_count += len(batch)
                    print(f"      ... Injected {total_count} concepts ({(time.time() - start_time):.2f}s)")
                    batch = []
            
            # Final batch
            if batch:
                cursor.executemany("""
                    INSERT OR IGNORE INTO nodes (id, name, definition, tags, frequency, created_at, realm, gravity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                total_count += len(batch)
                
        except Exception as e:
            print(f"  Error during massive injection: {e}")
                
        except Exception as e:
            print(f"  Error during massive injection: {e}")
        finally:
            conn.close()
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"  Massive Genesis Complete.")
        print(f"   Total New Concepts: {total_count}")
        print(f"   Time Elapsed: {duration:.2f}s")
        print(f"   Injection Rate: {total_count / duration:.0f} concepts/sec")
        print(f"   Current Vocabulary Size: {self.memory.get_concept_count()}")

if __name__ == "__main__":
    genesis = MassiveFractalGenesis()
    genesis.inject_massive_fractals()
