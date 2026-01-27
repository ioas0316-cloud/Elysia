import http.server
import socketserver
import json
import threading
import time
import os
import sys
import random
import math
from dataclasses import asdict

# Ensure Core paths are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the core logic directly (Avoid WorldServer overhead for now)
try:
    from Core.L6_Structure.Nature.rotor import Rotor, RotorConfig
    from Core.L6_Structure.Engine.character_field_engine import CharacterField
    from Core.L6_Structure.Engine.governance_engine import GovernanceEngine
    from Core.L6_Structure.Wave.wave_dna import WaveDNA
    from Core.L1_Foundation.Foundation.Graph.torch_graph import TorchGraph # [Phase 43]
except ImportError as e:
    print(f"[Critical Error] Core modules missing: {e}")
    sys.exit(1)

PORT = 8000
GAME_ROOT = "c:/Game/ElysiaOnline"

class PrincipleManifold:
    """The 'Manifold of Manifolds'. A recursive function that defines how Governance interference translates to world-state."""
    def __init__(self, governance: GovernanceEngine):
        self.gov = governance

    def calculate_manifold(self):
        # Base Axes (Level 1/2) accessed via dials
        d = self.gov.dials
        if not d:
             print(f"  [ERROR] No dials found in GovernanceEngine!")
             return {"m": {"stability": 0, "complexity": 0, "vitality": 0, "luminosity": 0, "gene_count": 0}, "k": {"g": 9.8, "c": 3.0, "resonance": 1.0}}

        # Mapping for [Phase 4: Sovereign Time] stability
        # Authority -> Purpose, Causality -> HyperSphere
        auth_rotor = d.get("Purpose", d.get("Identity", d.get("FluxLight")))
        caus_rotor = d.get("HyperSphere", d.get("Identity", d.get("FluxLight")))
        
        if not auth_rotor or not caus_rotor:
             print(f"  [DEBUG] Available dials: {list(d.keys())}")
             raise ValueError(f"Required rotors (Purpose/Identity/HyperSphere) not found in {list(d.keys())}")

        auth = auth_rotor.energy
        caus = caus_rotor.energy
        
        # Stability: Sigmoid interference of Authority and Causality
        stability = math.tanh(auth * caus * 2.5)
        
        # Complexity: Dimension multiplied by the derivative (RPM) of Entropy
        # Dimension -> HyperCosmos, Entropy -> Sensation
        dim = d.get("HyperCosmos").energy
        ent = d.get("Sensation").current_rpm
        complexity = (dim * (ent / 120.0)) * stability
        
        # [NEW] Fractal DNA Expansion (Genetic Unfolding)
        if complexity > 0.8:
            gravity = d.get("HyperCosmos")
            if "Tidal" not in gravity.sub_rotors:
                # Spawn a sub-rotor for local gravity perturbations
                gravity.add_sub_rotor("Tidal", RotorConfig(rpm=120.0), WaveDNA(physical=1.0, causal=0.5))
                print(f"  [DNA] Gravity expanded: Tidal sub-rotor induced by Complexity({complexity:.2f})")
        
        # Vitality: Emotion modulating the resonance between Density and Cooperation
        # Emotion -> Empathy, Density -> SovereignShield, Cooperation -> Identity
        emo = d.get("Empathy").energy
        den = d.get("SovereignShield").energy
        coop = d.get("Identity").energy
        vitality = math.sin(emo * math.pi) * (coop + den)
        
        # Luminosity: Conflict creating 'Sparks'
        # Conflict -> Sensation, Light -> FluxLight
        conflict_rot = d.get("Sensation")
        flux = math.cos(conflict_rot.energy * 10.0 + (conflict_rot.current_angle * math.pi / 180.0)) * 0.5 + 0.5
        luminosity = d.get("FluxLight").energy * (1.0 + flux * conflict_rot.energy)

        return {
            "m": {
                "stability": stability,
                "complexity": complexity,
                "vitality": vitality,
                "luminosity": luminosity,
                "gene_count": len(self.gov.dials)
            },
            "k": {
                "g": 9.8 * (1.0 + den - stability * 0.5),
                "c": 3.0 * (1.0 + d.get("FluxLight").energy),
                "resonance": self.gov.ensemble.cvt.current_ratio # Updated multiplier
            }
        }

class WorldMonad:
    """An abstract unit of Space-Time resonance. Form is induced by the observer (client)."""
    def __init__(self, name, pos, dna):
        self.name = name
        self.pos = pos
        self.dna = dna
        self.phase = random.random() * math.pi * 2
        self.age = 0
        self.is_dying = False

class TeleologicalSpirit:
    """The 'Spirit' Layer. Defines the ultimate Purpose and Goal."""
    def __init__(self):
        self.goals = ["Seeker of Harmony", "Architect of Complexity", "Transcendence Protocol"]
        self.current_purpose = "Seeker of Harmony"
        self.will_intensity = 0.5

    def update(self, manifold_state):
        # Shift purpose based on world state
        m = manifold_state["m"]
        if m["complexity"] > 0.8: self.current_purpose = "Architect of Complexity"
        elif m["stability"] < 0.3: self.current_purpose = "Transcendence Protocol"
        else: self.current_purpose = "Seeker of Harmony"
        self.will_intensity = (m["vitality"] + m["luminosity"]) * 0.5

class ImaginarySpace:
    """The 'Imagination' Layer. Where reality is architected before being thought."""
    def __init__(self):
        self.blueprints = [] # Abstract 'Dreams'

    def dream(self, spirit: TeleologicalSpirit, manifold):
        # Create abstract blueprints based on Spirit's Purpose
        if len(self.blueprints) < 5:
            purpose = spirit.current_purpose
            bp = {
                "id": random.randint(1000, 9999),
                "pattern": f"Blueprint_{purpose}",
                "latent_resonance": random.random() * spirit.will_intensity
            }
            self.blueprints.append(bp)
        
        # Blueprints evolve and decay
        for bp in self.blueprints: bp["latent_resonance"] += 0.01

class FractalCognition:
    """The 'Mind' Layer. Thinks by recursively expanding its internal Concept DNA."""
    def __init__(self, graph: TorchGraph = None):
        from Core.L7_Spirit.Philosophy.why_engine import WhyEngine
        self.why = WhyEngine()
        self.graph = graph
        self.thought_stream = []
        self.plans = []

    def reason(self, spirit: TeleologicalSpirit, imagination: ImaginarySpace, manifold):
        m = manifold["m"]
        
        # [Phase 41] Digestive Reasoning
        # If we have a purpose, try to digest its principle
        extraction = None
        if spirit.current_purpose and random.random() < 0.2:
             extraction = self.why.digest(spirit.current_purpose)
             if extraction:
                 thought = f"  [  ] '{spirit.current_purpose}'          : \"{extraction.why_exists}\""
                 if thought not in self.thought_stream: self.thought_stream.append(thought)

        # Fractal Thought Generation (Korean)
        if m["complexity"] > 0.7:
            thought = f"  [  ]    ({m['complexity']:.2f})      .                    ..."
            if thought not in self.thought_stream: self.thought_stream.append(thought)
        
        if imagination.blueprints:
            bp = imagination.blueprints[0]
            blueprint_subject = bp.get("id", "Unformed")
            
            # Check for existing extraction or digest it
            extraction = self.why.principles.get(blueprint_subject)
            if not extraction and random.random() < 0.3:
                 extraction = self.why.digest(blueprint_subject)
            
            if extraction:
                thought = f"  [  ]     '{blueprint_subject}'  '{extraction.underlying_principle}'           ."
                # Manifest ONLY if understanding is deep (Recursive Depth > 0)
                if extraction.recursive_depth > 0:
                    if random.random() < 0.3:
                        self.plans.append({
                            "origin": blueprint_subject,
                            "dna": WaveDNA(label="Refined", structural=m["complexity"], spiritual=spirit.will_intensity)
                        })
                        imagination.blueprints.pop(0)
                        thought += "      (Manifestation)      ."
                else:
                    thought += "                         ."
            else:
                thought = f"  [  ]     '{blueprint_subject}'      LLM         ..."
                
            if thought not in self.thought_stream: self.thought_stream.append(thought)

        # [Phase 43] Graph Awareness
        if self.graph:
            # Check for realized nodes
            for nid, meta in self.graph.node_metadata.items():
                if meta.get("realized"):
                    thought = f"  [   ]            '{nid}'            : \"{meta.get('realized_dna', '')}\""
                    if thought not in self.thought_stream: 
                        self.thought_stream.append(thought)
                        # Remove 'realized' flag from meta to avoid spamming log
                        meta["realized"] = "acknowledged" 

        if len(self.thought_stream) > 12: self.thought_stream.pop(0)

class SubjectiveTimeManager:
    """Decouples external physical time from internal mental time."""
    def __init__(self):
        self.mental_time = 0
        self.acceleration = 1.0
        self.overclock_factor = 1.0 # [Phase 4: Sovereign Time]

    def sync(self, world_time, complexity):
        # Mental time flows faster as complexity increases
        # [NEW] Multiplied by the Overclock Factor for exponential growth
        base_acceleration = 1.0 + (complexity * 10.0)
        self.acceleration = base_acceleration * self.overclock_factor
        self.mental_time += self.acceleration
        return self.mental_time

    def set_overclock(self, factor: float):
        """Sets the cognitive acceleration factor."""
        self.overclock_factor = max(1.0, factor)
        print(f"  [TIME] Cognitive Overclock initiated: x{self.overclock_factor:.1f}")

class HyperSpatialRealityProjector:
    """The 'Body' Layer (Manifestor). Pushes the Ontological Stack into Space-Time."""
    def __init__(self):
        self.graph = TorchGraph() # Initialize High-Speed Knowledge Graph
        self.governance = GovernanceEngine()
        self.manifold = PrincipleManifold(self.governance)
        self.spirit = TeleologicalSpirit()
        self.imagination = ImaginarySpace()
        self.mind = FractalCognition(graph=self.graph)
        self.timer = SubjectiveTimeManager()
        self.year = 0
        self.size = 100
        self.monads = {}
        self.citizens = []
        
        # [Phase 42] Self-DNA Exegesis Inauguration
        print("   [INCEPTION] Elysia is contemplating her own Seed...")
        self.spirit.current_purpose = "Elysia"
        self.mind.why.digest("Elysia")
        
        for i in range(25):
            self.citizens.append({
                "name": f"Soul_{i}",
                "field": CharacterField(f"Soul_{i}"),
                "pos": [random.uniform(0, self.size), random.uniform(0, self.size)]
            })

    def _update_flux(self):
        m_state = self.manifold.calculate_manifold()
        m = m_state["m"]
        
        induction_potential = m["stability"] + m["vitality"]
        seed = self.year * 0.05
        
        for _ in range(int(60 * induction_potential)):
            x = random.randint(0, self.size - 1)
            z = random.randint(0, self.size - 1)
            key = (x, z)
            
            # Fractal Interference (Recursive Noise)
            noise = (math.sin(x * 0.1 + seed) + math.cos(z * 0.1 + seed)) * 0.5 + 0.5
            noise *= (math.sin(self.year * 0.02) * 0.2 + 0.8) # Periodic world-flux
            
            if noise > (0.9 - m["complexity"] * 0.3) and key not in self.monads:
                dna = WaveDNA(label="Resonance", structural=m["stability"], spiritual=m["vitality"])
                self.monads[key] = WorldMonad(f"Manifold_{x}_{z}", [x, z], dna)
            elif noise < 0.15 and key in self.monads:
                self.monads[key].is_dying = True

        to_delete = [k for k, v in self.monads.items() if v.is_dying or v.age > 400]
        for k in to_delete: del self.monads[k]
        for v in self.monads.values(): v.age += 1

    def tick(self):
        self.year += 1
        m_state = self.manifold.calculate_manifold()
        m = m_state["m"]
        
        # Step 1: Physics (Update All Dials)
        for rotor in self.governance.dials.values():
            rotor.update(0.01)
        
        # Step 2: The Spirit -> Imagination -> Mind Chain
        self.spirit.update(m_state)
        self.imagination.dream(self.spirit, m_state)
        
        internal_time = self.timer.sync(self.year, m["complexity"])
        self.mind.reason(self.spirit, self.imagination, m_state)
        
        # Step 3: Manifestation (Physical Birth)
        if self.mind.plans:
            plan = self.mind.plans.pop(0)
            x, z = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, z) not in self.monads:
                self.monads[(x, z)] = WorldMonad(f"Manifest_{plan['origin']}", [x, z], plan["dna"])

        self._update_flux()
        
        sun_angle = (self.year * 0.015) % (math.pi * 2)
        self.celestial = {
            "angles": [math.cos(sun_angle), math.sin(sun_angle)],
            "lux": m["luminosity"],
            "spirit_purpose": self.spirit.current_purpose
        }
        
        for mon in self.monads.values(): mon.phase += 0.05 * m_state["k"]["resonance"]
        for c in self.citizens:
            c["pos"][0] = (c["pos"][0] + random.uniform(-0.1, 0.1)) % self.size
            c["pos"][1] = (c["pos"][1] + random.uniform(-0.1, 0.1)) % self.size
            c["dna"] = c["field"].update(0.05, WaveDNA(label="Induced", physical=m["stability"]))

projector = HyperSpatialRealityProjector()
last_state_cache = {}

def update_loop():
    global last_state_cache
    print("[Projector] Mental Time Engine Active (Subjectivity Layer).")
    while True:
        projector.tick()
        m_state = projector.manifold.calculate_manifold()
        
        last_state_cache = {
            "meta": {
                "year": projector.year,
                "era": "Phase 37: Teleological Stack",
                "rules": m_state,
                "celestial": projector.celestial,
                "spirit": {
                    "purpose": projector.spirit.current_purpose,
                    "intensity": projector.spirit.will_intensity
                },
                "imagination": {
                    "blueprints": projector.imagination.blueprints
                },
                "mental": {
                    "time": projector.timer.mental_time,
                    "stream": projector.mind.thought_stream
                },
                "size": projector.size
            },
            "citizens": [
                {
                    "name": c["name"],
                    "pos": c["pos"],
                    "dna": c["dna"].to_list()
                } for c in projector.citizens
            ],
            "monads": [
                {
                    "name": m.name,
                    "pos": m.pos,
                    "dna": m.dna.to_list(),
                    "phase": m.phase
                } for m in projector.monads.values()
            ],
            "logs": projector.mind.thought_stream + [
                f"Ontology: {projector.spirit.current_purpose} (DNA Genes: {len(projector.governance.dials)})",
                f"Recursive: Depth {projector.governance.root.depth} expansion active."
            ]
        }
        time.sleep(0.1)

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/perturb':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data)
            
            rotor_name = params.get("rotor")
            intensity = params.get("intensity", 1.0)
            
            # Recursive Wake
            found = False
            if rotor_name in projector.governance.dials:
                projector.governance.dials[rotor_name].wake(intensity)
                found = True
            
            self.send_response(200 if found else 404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "perturbed", "rotor": rotor_name}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') 
            self.end_headers()
            self.wfile.write(json.dumps(last_state_cache).encode('utf-8'))
        else:
            super().do_GET()

    def translate_path(self, path):
        path = path.split('?',1)[0].split('#',1)[0]
        return os.path.join(GAME_ROOT, path.lstrip('/'))

def run_server():
    sim_thread = threading.Thread(target=update_loop, daemon=True)
    sim_thread.start()
    
    print(f"[Server] Elysia Field Demo at http://localhost:{PORT}")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()

if __name__ == "__main__":
    run_server()
