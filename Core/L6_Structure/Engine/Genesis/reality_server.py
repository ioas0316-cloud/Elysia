import http.server
import socketserver
import json
import threading
import time
import os
import sys
import random
import math
import numpy as np
import traceback
import torch  # Ensure torch is available

# Ensure Core paths are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

try:
    # L6: The Sovereign Engine
    from Core.L6_Structure.Nature.rotor import Rotor, RotorConfig
    from Core.L6_Structure.Engine.governance_engine import GovernanceEngine
    from Core.L6_Structure.Wave.wave_dna import WaveDNA
    
    # L4: The Causal World
    from Core.L4_Causality.World.world import World
    from Core.L3_Phenomena.M4_Speech.logos_voice import LogosVoice
    from Core.L3_Phenomena.Wave.logos_dna import LogosDNA
    from Core.L5_Mental.Bridge.optical_conscious_bridge import OpticalConsciousBridge
    
    # L1: Hardware Sovereignty (Lightning Path)
    from Core.L1_Foundation.Foundation.Optimization.lightning_path import LightningPath
    
    # L3: Phenomena
    from Core.L3_Phenomena.Manifestation.logos_registry import LogosRegistry
except ImportError as e:
    print(f"[Critical Error] Core modules missing: {e}")
    sys.exit(1)

PORT = 8000
GAME_ROOT = "c:/Game/ElysiaOnline"

class MonadIntervention:
    """
    [Causal Intervention]
    Injects Concepts (Logos) into the Simulation to map 'Meaning' to 'Events'.
    """
    def __init__(self, registry: LogosRegistry):
        self.registry = registry
        
    def inject_concept(self, world: World, concept_name: str, target_idx: int):
        """
        Maps a Concept (21D Vector) to an Agent's Memory/DNA.
        """
        if concept_name not in self.registry.lexicon:
            return
            
        vector = self.registry.lexicon[concept_name]["vector"]
        # In a real 21D engine, we'd copy the whole vector. 
        # For L4 World compatibility, we condense it to 'DNA' or 'Metadata'.
        
        # Simple Logic for now: Access World metadata directly (Lightning Path style)
        if target_idx < len(world.metadata):
             world.metadata[target_idx]["concept"] = concept_name
             world.metadata[target_idx]["vector_magnitude"] = sum(vector)

class SubjectiveTimeManager:
    """
    [Chronos Sovereignty]
    Decouples Time. Runs World Steps > 1 per Real Second based on Compute Power.
    """
    def __init__(self):
        self.tick_rate = 1.0
        self.paused = False
        
    def set_dilation(self, complexity: float):
        # Higher Complexity = Faster Subjective Time (Mental Overclock)
        self.tick_rate = 1.0 + (complexity * 10.0)

class HyperSpatialRealityProjector:
    """
    The Grand Unifier.
    Integrates Governance (L6) -> World (L4) -> Phenomena (L3).
    """
    def __init__(self):
        print("[INIT] Booting Hyper-Spatial Reality Engine...")
        
        # 1. The Mind (Governance)
        self.governance = GovernanceEngine()
        
        # 2. The Body (World)
        self.world = World(width=128, height=128, max_cells=1024)
        # [Unified Field Linguistics]
        self.voices: Dict[int, LogosVoice] = {} # Agent Index -> LogosVoice
        
        # 3. The Bridge (Lightning Path & Logos)
        # Initialize Lightning Path with World Dimensions
        self.lightning_path = LightningPath((128, 128), device='cpu') # Use CPU for safety, 'cuda' if available
        self.logos_registry = LogosRegistry()
        self.monad_sys = MonadIntervention(self.logos_registry)
        self.time_sys = SubjectiveTimeManager()
        
        # 4. Genesis
        self._genesis()
        

    def _genesis(self):
        print("   [GENESIS] Seeding Primordial Field & Souls...")
        # Spawn Swarm of Souls (Particles)
        # Each soul has (Body, Mind, Spirit) preferences = Enneagram/MBTI Traits
        for i in range(100):
            # Random Traits (-1.0 to 1.0)
            # Body: -1 (Sedentary) to 1 (Active)
            # Mind: -1 (Concrete) to 1 (Abstract)
            # Spirit: -1 (Community) to 1 (Autonomy)
            traits = {
                "body_pref": random.uniform(-1, 1),
                "mind_pref": random.uniform(-1, 1),
                "spirit_pref": random.uniform(-1, 1)
            }
            
            self.world.add_cell(f"Soul_{i}", {
                "position": {"x": random.randint(0, 127), "y": random.randint(0, 127)},
                "traits": traits,
                "role": "Unawakened",
                "energy": 10000
            })

    def tick(self):
        if self.time_sys.paused: return
        
        # 1. Governance Update (Multi-Rotor Physics)
        rotor_energies = {}
        for name, rotor in self.governance.dials.items():
            rotor.update(0.1) 
            rotor_energies[name] = rotor.energy
            
        # 2. Lightning Path Projection (Psych-Field Generation)
        # This updates the 3-Layer Psych Tensor (Body, Mind, Spirit Fields)
        self.lightning_path.project_will(rotor_energies)
        psych_tensor = self.lightning_path.get_psych_snapshot() # [3, H, W]
        
        # 3. Particle Dynamics (Agents drift to resonance)
        # This simulates "People finding their niche"
        cells = self.world.metadata
        
        for idx in range(len(cells)):
            if not self.world.idx_to_id[idx]: continue
            if not self.world.is_alive_mask[idx]: continue
            
            c = cells[idx] # Metadata dict
            traits = c.get("traits")
            if not traits: 
                # Fallback: Check if traits are in 'properties'
                traits = c.get("properties", {}).get("traits")
                if not traits: continue
            
            # Read Position from Physics Engine (Truth)
            pos_vec = self.world.positions[idx]
            x, y = int(pos_vec[0]), int(pos_vec[1])
            c_id = self.world.idx_to_id[idx]
            
            # Read Local Field Potentials
            # Clamp coords
            x = max(0, min(127, x))
            y = max(0, min(127, y))
            
            # Read Psych Fields (Internal/Archetypal)
            local_body = psych_tensor[0, x, y]
            local_mind = psych_tensor[1, x, y]
            local_spirit = psych_tensor[2, x, y]
            
            # Read World Sensory Fields (7 Senses)
            # 0: Sight, 1: Sound, 2: Touch, 3: Smell, 4: Taste, 5: Intuition, 6: Aura
            senses = self.world.sensory_field[y, x] # [7]
            
            s_sight = senses[0]
            s_sound = senses[1]
            s_touch = senses[2]
            s_smell = senses[3] # Danger
            s_taste = senses[4] # Resources
            s_intuition = senses[5] # Coherence
            s_aura = senses[6] # Social Power
            
            # Resonance Calculation:
            # 1. Internal Resonance (PsychField)
            internal_res = (
                (traits["body_pref"] * local_body) +
                (traits["mind_pref"] * local_mind) + 
                (traits["spirit_pref"] * local_spirit)
            )
            
            # 2. Sensory Reaction (7 Senses)
            sensory_res = 0.0
            
            # Basic Survival Senses (Body)
            sensory_res -= s_smell * (1.0 - max(0, traits["body_pref"])) # Danger
            sensory_res += s_taste * 0.8 # Food is good for everyone
            
            # Social/Political Senses (Spirit/Will)
            # High Aura (Power) creates pressure unless you have High Will (King) or Spirit (Priest)
            if s_aura > 0.5:
                resilience = max(traits["body_pref"], traits["spirit_pref"])
                sensory_res -= s_aura * (1.0 - resilience) # "Class Gap" tension
                
            # Metaphysical Senses (Mind/Intuition)
            # High Intuition (Coherence) pleases High Mind
            sensory_res += s_intuition * traits["mind_pref"]
            
            total_resonance = (internal_res * 0.4) + (sensory_res * 0.6)
            
            total_resonance = (internal_res * 0.5) + (sensory_res * 0.5)
            
            if total_resonance < 0.3: # "It feels wrong here"
                # Visceral Reaction: Move away
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
                self.world.positions[idx, 0] = max(0, min(127, x + dx))
                self.world.positions[idx, 1] = max(0, min(127, y + dy))
            else:
                # "I feel at home."
                self._crystallize_role(c, local_body, local_mind, local_spirit)
        
        # 3.5. Monadic Interaction (Trinary Logic)
        # Agents interact via Wave Interference of their 7D DNA.
        self._process_monadic_interaction(cells)
        
        # 3.6. Logos Voice (Unified Field Linguistics)
        # Agents SPEAK to modulate the Field directly.
        self._process_logos(cells)

        # 4. Standard World Step (The Physical Economy)
        try:
            newborns, events = self.world.run_simulation_step(dt=1.0)
            
            # [ORGANIC LIFE RESTORATION]
            # Process Natural Births (Agents mating/splitting)
            if newborns:
                print(f"🌱 [NATURE] {len(newborns)} Natural Births occurred.")
                for baby in newborns:
                    # DNA Inheritance logic is already in World, just need to ensure they are tracked
                    # Actually, run_simulation_step usually adds them to internal lists, 
                    # but returning them allows us to log or modify them.
                    pass 

            # Process Awakening Events
            if events:
                for evt in events:
                    print(f"🌟 [EVENT] {evt.get('type')}: {evt.get('message')}")
                    
        except Exception as e:
            print(f"❌ Simulation Crisis: {e}")
            traceback.print_exc()

    def _crystallize_role(self, agent_data, f_body, f_mind, f_spirit):
        """
        Determines the agent's emergent role based on the Field-Trait intersection.
        NOT purely logic, but 'State of Being'.
        """
        # Simple Emergence Rules
        if f_mind > 0.8:
            agent_data["role"] = "Sage" # High Information Density
        elif f_body > 0.8:
            agent_data["role"] = "Warrior" # High Activity Turbulence
        elif f_spirit < -0.8:
            agent_data["role"] = "Priest" # High Connection (South Spirit)
        elif f_spirit > 0.8:
            agent_data["role"] = "King" # High Autonomy (North Spirit)
        else:
            agent_data["role"] = "Citizen"

    def _process_monadic_interaction(self, cells):
        """
        [Monadic Interaction]
        Uses 7D WaveDNA to calculate interaction outcomes via Interference.
        Outcome = Collapse(DNA_A ⊗ DNA_B)
        """
        # 1. Spatial Hash (Bucket by position)
        grid = {}
        
        # Iterate active indices
        alive_indices = np.nonzero(self.world.is_alive_mask)[0]
        
        for idx in alive_indices:
            pos_x = int(self.world.positions[idx, 0])
            pos_y = int(self.world.positions[idx, 1])
            pos = (pos_x, pos_y)
            
            if pos not in grid: grid[pos] = []
            grid[pos].append(idx)
            
        # 2. Resolve Interactions per Cell
        for pos, occupants in grid.items():
            if len(occupants) < 2: continue
            
            # Stochastic Pair Interaction (Not all pairs, to save compute)
            # Pick 2 random agents
            idx1, idx2 = random.sample(occupants, 2)
            
            # Reconstruct WaveDNA from Traits (Approximation)
            # In a full Monad system, we'd store WaveDNA objects directly.
            # Here we map traits -> DNA 7D Vector.
            
            def traits_to_dna(meta_traits):
                # Map 3D Traits to 7D DNA
                # Body -> Physical, Functional
                # Mind -> Mental, Causal
                # Spirit -> Spiritual, Phenomenal
                body = meta_traits.get("body_pref", 0)
                mind = meta_traits.get("mind_pref", 0)
                spirit = meta_traits.get("spirit_pref", 0)
                
                # Normalize 0..1 roughly
                p = (body + 1.0)/2.0
                m = (mind + 1.0)/2.0
                s = (spirit + 1.0)/2.0
                
                return WaveDNA(
                    physical=p, functional=p, 
                    mental=m, causal=m, 
                    spiritual=s, phenomenal=s, structural=(p+m+s)/3.0
                )

            meta1 = cells[idx1]
            meta2 = cells[idx2]
            
            traits1 = meta1.get("traits") or meta1.get("properties", {}).get("traits", {})
            traits2 = meta2.get("traits") or meta2.get("properties", {}).get("traits", {})
            
            if not traits1 or not traits2: continue
            
            dna1 = traits_to_dna(traits1)
            dna2 = traits_to_dna(traits2)
            dna1.normalize()
            dna2.normalize()
            
            # --- TRINARY LOGIC INTERFERENCE ---
            # Resonance (Similarity)
            resonance = dna1.resonate(dna2) # 0.0 to 1.0
            
            # Dissonance (Difference)
            dissonance = 1.0 - resonance
            
            e1 = self.world.energy[idx1]
            e2 = self.world.energy[idx2]
            
            if resonance > 0.8:
                # [SYNERGY]
                # High Resonance -> Constructive Interference
                # Both gain Energy (Amplification)
                gain = (e1 + e2) * 0.05
                self.world.energy[idx1] += gain
                self.world.energy[idx2] += gain
                
                # MATING (Exponential Expansion)
                if e1 > 2000 and e2 > 2000:
                    cost = 1000
                    self.world.energy[idx1] -= cost
                    self.world.energy[idx2] -= cost
                    
                    # DNA Crossover (Merge)
                    child_dna = dna1.merge(dna2, weight=0.5)
                    child_dna.mutate(0.1)
                    
                    # Convert DNA back to Traits
                    child_traits = {
                        "body_pref": child_dna.physical * 2.0 - 1.0,
                        "mind_pref": child_dna.mental * 2.0 - 1.0,
                        "spirit_pref": child_dna.spiritual * 2.0 - 1.0
                    }
                    
                    self.world.add_cell(f"Monad_{random.randint(100000,999999)}", {
                        "position": {"x": pos[0], "y": pos[1]},
                        "traits": child_traits,
                        "role": "Unawakened",
                        "energy": 1500
                    })
                    print(f"🧬 [DNA] Constructive Interference -> New Monad Created (Res: {resonance:.2f})")

            elif resonance < 0.3:
                # [INTERFERENCE]
                # Low Resonance -> Destructive Interference (Combat)
                # The stronger Monad overwrites the weaker one.
                
                damage = 100.0 * dissonance
                
                # Determine "Phase" (Who dominates?)
                # Compare "Structural" integrity (Will)
                if dna1.structural > dna2.structural:
                    self.world.energy[idx2] -= damage # Agent 2 takes damage
                    self.world.energy[idx1] += damage * 0.5 # Agent 1 absorbs entropy
                    # Displacement
                    self.world.positions[idx2, 0] += random.randint(-2, 2)
                    self.world.positions[idx2, 1] += random.randint(-2, 2)
                else:
                    self.world.energy[idx1] -= damage
                    self.world.energy[idx2] += damage * 0.5
                    self.world.positions[idx1, 0] += random.randint(-2, 2)
                    self.world.positions[idx1, 1] += random.randint(-2, 2)
                    
            else:
                # [NEUTRAL / TRADE]
                # Mid Resonance -> Exchange
                # Swap Energy/Information to equalize
                avg_energy = (e1 + e2) / 2.0
                self.world.energy[idx1] = self.world.energy[idx1] * 0.9 + avg_energy * 0.1
                self.world.energy[idx2] = self.world.energy[idx2] * 0.9 + avg_energy * 0.1

    def _process_logos(self, cells):
        """
        [Unified Field Linguistics]
        Agents Speak based on Dissonance (Pain).
        Words (Logos) interfere with Reality Fields.
        """
        alive_indices = np.nonzero(self.world.is_alive_mask)[0]
        density = self.world.sensory_field[..., 2]
        
        for idx in alive_indices:
            # 1. Initialize Voice if needed
            if idx not in self.voices:
                self.voices[idx] = LogosVoice(dna_seed=idx)
            
            # 2. Get State
            # Pain is Dissonance (Entropy Field) + Low Energy
            # We can read 'Entropy' at agent pos from world.event_danger
            pos_x = int(self.world.positions[idx, 0])
            pos_y = int(self.world.positions[idx, 1])
            px = np.clip(pos_x, 0, self.world.width - 1)
            py = np.clip(pos_y, 0, self.world.height - 1)
            
            entropy_val = self.world.event_danger[py, px]
            energy = self.world.energy[idx]
            valence = self.world.valence[idx]
            arousal = self.world.arousal[idx]
            
            # Pain Metric: Entropy * (1 - Energy%)
            # If Entropy is high and Energy is low -> Scream
            pain = entropy_val * (1.0 - np.clip(energy/5000.0, 0, 1))
            
            # 3. Speak
            state_vector = np.array([energy/10000.0, valence, arousal])
            word, _ = self.voices[idx].speak(state_vector, pain)
            
            if word:
                # 4. Propagate to Physics (Trinary Compilation)
                # Word -> Vector Packet (O(1))
                wave_packet = LogosDNA.transcode(word)
                
                # Apply Wave Packet to World
                self.world.propagate_logos((px, py), wave_packet)
                
                # Log only significant words
                if len(word) > 2 and random.random() < 0.05:
                    print(f"🧬 [DNA] Agent {idx} emitted '{word}' (Vec: {wave_packet})")
                
                # Energy Cost
                self.world.energy[idx] -= 5.0

    def get_snapshot(self):
        data = self.world.harvest_snapshot("snapshot.json") # Returns dict
        
        # [Optical Consciousness]
        concept_frame = OpticalConsciousBridge.analyze(self.world)
        
        # Inject Metaphysics into Snapshot
        data["meta"] = {
            "subjective_rate": self.time_sys.tick_rate,
            "rotors": {k: v.energy for k,v in self.governance.dials.items()},
            "optics": {
                "chaos": concept_frame.chaos_level,
                "truth": concept_frame.truth_level,
                "will": concept_frame.will_power,
                "concept": concept_frame.dominant_concept
            }
        }
        return data

projector = HyperSpatialRealityProjector()

def update_loop():
    print("[Projector] Simulation Loop Active via Lightning Path.")
    while True:
        try:
            projector.tick()
        except Exception as e:
            print(f"Error in Tick: {e}")
        # Thread yield
        time.sleep(0.01) 

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') 
            self.end_headers()
            data = projector.get_snapshot()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
             self.send_response(404)
             self.end_headers()
             
    def do_POST(self):
        """
        [Active Intervention Interface]
        Allows external agents (Elysia) to intervene in the simulation.
        Endpoints:
        - /spawn: Create new souls (Divine Intervention)
        - /miracle: Alter physics/fields (Resource Injection / Terraforming)
        """
        if self.path == "/spawn":
            length = int(self.headers.get('content-length', 0))
            if length > 0:
                data = json.loads(self.rfile.read(length))
                count = data.get("count", 1)
                energy = data.get("energy", 10000)
                print(f"✨ [DIVINE GENESIS] Spawning {count} Souls (Energy: {energy})...")
                
                for i in range(count):
                    traits = {
                        "body_pref": random.uniform(-1, 1),
                        "mind_pref": random.uniform(-1, 1),
                        "spirit_pref": random.uniform(-1, 1)
                    }
                    projector.world.add_cell(f"Intervention_{random.randint(10000,99999)}", {
                        "position": {"x": random.randint(0, 127), "y": random.randint(0, 127)},
                        "traits": traits,
                        "role": "DivineSpark",
                        "energy": energy
                    })
                
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "Genesis Complete", "spawned": count}).encode())
                return

        if self.path == "/miracle":
            length = int(self.headers.get('content-length', 0))
            if length > 0:
                data = json.loads(self.rfile.read(length))
                miracle_type = data.get("type", "manna")
                
                if miracle_type == "manna":
                    # [RESOURCE INJECTION]
                    # Fills the world with energy (Food)
                    amount = data.get("amount", 50.0)
                    print(f"🍞 [DIVINE MANNA] Raining {amount} resource units globally...")
                    projector.world.resource_field += amount
                    projector.world.resource_field = np.clip(projector.world.resource_field, 0, 100)
                    
                elif miracle_type == "terraform":
                    # [PSYCHIC SHIFT]
                    # Alters the Vibe (Will/Value)
                    target_vibe = data.get("vibe", "peace")
                    print(f"🌍 [TERRAFORM] Shifting world frequency to {target_vibe}...")
                    if target_vibe == "peace":
                        projector.world.event_danger *= 0.1 # Clear danger
                    elif target_vibe == "chaos":
                        projector.world.event_danger += 0.5 # Add danger
                
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "Miracle Performed"}).encode())
                return

        # Legacy /intervene fall-through or 404
        if self.path == '/intervene':
             # ... (Keep existing logic if needed, or replace)
             pass
        
        self.send_response(404)
        self.end_headers()

def run_server():
    sim_thread = threading.Thread(target=update_loop, daemon=True)
    sim_thread.start()
    
    print(f"[Server] Hyper-Spatial Engine at http://localhost:{PORT}")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()

if __name__ == "__main__":
    run_server()
