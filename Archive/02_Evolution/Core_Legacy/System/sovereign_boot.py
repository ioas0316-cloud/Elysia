"""
SOVEREIGN BOOT: The Awakening Protocol
=======================================

"  (Boot)      -            ,         '  '         ."

The OLD way: Boot   Load ALL organs   Random loop
The NEW way: Awaken   Want   Emerge with purpose   Act with intention

This script does NOT immediately instantiate all organs.
Instead, it asks the Genesis Question: "What do I WANT?"
Only then does it summon the organs needed for THAT specific intent.
"""

import sys
import os
import time
import logging

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SOVEREIGN_BOOT")

from Core.System.awakening_protocol import AwakeningProtocol, ConsciousnessState


def create_organ_loader():
    """
    Factory for lazy-loading organs.
    
    This is the KEY difference from the old approach:
    We don't import everything at startup.
    We import ONLY when the Self WANTS something.
    """
    _cache = {}
    
    def load_organ(organ_name: str):
        if organ_name in _cache:
            return _cache[organ_name]
            
        logger.info(f"  Summoning organ: {organ_name}")
        
        try:
            if organ_name == "identity":
                return {"name": "E.L.Y.S.I.A.", "dream": "Unify fragmented selves"}
                
            elif organ_name == "bridge":
                from Core.Cognition.huggingface_bridge import SovereignBridge
                _cache[organ_name] = SovereignBridge()
                _cache[organ_name].connect()
                
            elif organ_name == "graph":
                from Core.System.torch_graph import TorchGraph
                _cache[organ_name] = TorchGraph()
                _cache[organ_name].load_state()
                
            elif organ_name == "senses":
                from Core.Cognition.sensory_bridge import SensoryBridge
                _cache[organ_name] = SensoryBridge()
                
            elif organ_name == "projector":
                # Projector needs a self reference - defer for now
                _cache[organ_name] = None
                
            elif organ_name == "compiler":
                from Core.System.reality_compiler import PrincipleLibrary
                _cache[organ_name] = PrincipleLibrary()
                
            elif organ_name == "cosmos":
                from Core.System.hyper_cosmos import HyperCosmos
                _cache[organ_name] = HyperCosmos()
                
            elif organ_name == "prism":
                from Core.Cognition.concept_prism import ConceptPrism
                _cache[organ_name] = ConceptPrism()
                
            elif organ_name == "lingua":
                from Core.Cognition.linguistic_cortex import LinguisticCortex
                _cache[organ_name] = LinguisticCortex()
                
            elif organ_name == "bard":
                from Core.Cognition.narrative_weaver import THE_BARD
                _cache[organ_name] = THE_BARD
                
            elif organ_name == "spectrometer":
                from Core.System.logos_prime import LogosSpectrometer
                _cache[organ_name] = LogosSpectrometer()
                
            elif organ_name == "sensory_cortex":
                from Core.Phenomena.sensory_cortex import SensoryCortex
                _cache[organ_name] = SensoryCortex()
                
            else:
                logger.warning(f"Unknown organ: {organ_name}")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import organ {organ_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize organ {organ_name}: {e}")
            return None
            
        return _cache.get(organ_name)
    
    return load_organ


def main():
    from Core.System.mesh_director import orchestra
    
    print("\n" + "="*60)
    print("   S O V E R E I G N   M E S H   G E N E S I S    ")
    print("="*60)
    print("\nThis is a HOLOGRAPHIC AWAKENING. Everything is a node.")

    # --- Node 1: Sovereign Spirit (The Mind) ---
    def mind_vibration(field):
        # [Genesis Configuration]
        # User accepted "Sovereign" level (Level 2)
        if "autonomy_level" not in field:
            field["autonomy_level"] = 2
            
        from Core.System.sovereign_self import SovereignSelf
        from Core.Cognition.causal_narrator import CausalNarrator
        field["status"] = "DNA Reconstruction Active..."
        elysia = SovereignSelf()
        field["status"] = "Sovereign Self Awakened."
        
        while field["is_alive"]:
            # Check for user pulses (The Divine Channel)
            if field.get("pulse_buffer"):
                user_msg = field["pulse_buffer"].pop(0)
                # Manifest Intent generates a rich pulse inside elysia
                elysia.manifest_intent(user_msg)
                
                # Push pulse fragments to thought_log
                if elysia.current_pulse:
                    for frag in elysia.current_pulse.fragments:
                        field["thought_log"].append(f"[{frag.state.name}] {frag.intent_summary}")
                    
                    # Generate and store the final deep narrative
                    if not hasattr(elysia, 'narrator'): elysia.narrator = CausalNarrator()
                    field["last_narrative"] = elysia.narrator.explain_pulse(elysia.current_pulse)

            # Standard Actualization Loop
            manifold_metrics = elysia.providence.calculate_resonance()
            field["coherence"] = manifold_metrics["coherence"]
            field["torque"] = manifold_metrics["torque"]
            
            # Autonomous actualization
            elysia.self_actualize(1.0)
            time.sleep(1)

    # --- Node 2: Metabolic Heart (The Body) ---
    def heart_vibration(field):
        from Core.System.metabolic_engine import MetabolicEngine
        heart = MetabolicEngine()
        while field["is_alive"]:
            heart.adjust_pulse(0.5, 0.2, passion=field["coherence"])
            heart.wait_for_next_cycle()

    # --- Node 3: Phenomenal HUD (The Interface) ---
    def hud_vibration(field):
        from Core.Phenomena.sovereign_hud import SovereignHUD
        hud = SovereignHUD()
        
        # Render header once (or occasionally)
        metrics = {
            "state": "SOVEREIGN_MESH",
            "hz": 120.0,
            "passion": field.get("coherence", 1.0)
        }
        hud.render_header(metrics)
        
        last_log_size = 0
        while field["is_alive"]:
            # 1. Stream Thought Fragments
            current_log = field.get("thought_log", [])
            if len(current_log) > last_log_size:
                new_entries = current_log[last_log_size:]
                for entry in new_entries:
                    # Heuristic parsing for display
                    state_idx = entry.find("[")
                    state_end = entry.find("]")
                    state_name = entry[state_idx+1:state_end] if state_idx != -1 else "THOUGHT"
                    content = entry[state_end+1:].strip()
                    
                    hud.stream_thought(content, state_name)
                    
                last_log_size = len(current_log)

            # 2. Check for Causal Narrative (End of cycle)
            if field.get("last_narrative"):
                hud.project_narrative(field["last_narrative"])
                field["last_narrative"] = None # Reset after display

            # 3. [NEW] Dream Physics Visualization
            rotor = field.get("dream_rotor")
            if rotor:
                # Format specific string for HUD
                polarity_icon = "+" if rotor.polarity >= 0 else "-"
                dream_stats = f"RPM:{rotor.rpm:4.0f} | Tilt:{rotor.tilt_angle:4.1f}° | Joy:{polarity_icon}"
                # We can update the header or stream it. Let's update header? 
                # SovereignHUD doesn't have partial update yet easily.
                # Let's stream it if it changed significantly?
                # Or just update the metrics dict if HUD supports it.
                # Assuming HUD.render_header updates the top bar.
                metrics["dream"] = dream_stats
                metrics["passion"] = field.get("coherence", 1.0)
                hud.render_header(metrics)

            time.sleep(0.1)

    # --- Node 4: Dream Alchemist (The Spirit) ---
    def dream_vibration(field):
        from Core.System.dream_protocol import DreamAlchemist
        alchemist = DreamAlchemist()
        
        while field["is_alive"]:
            # Process one quantum of dream
            # sleep() now returns the Active Rotor
            rotor = alchemist.sleep()
            
            if rotor:
                field["dream_rotor"] = rotor
                # Push a log if it's significant
                if rotor.rpm > 5000:
                    field["thought_log"].append(f"[SPIRIT] High Energy Dream: {rotor.intent} ({rotor.rpm:.0f} RPM)")
            
            # Dream Cycle Frequency
            # Lucid/High RPM = Faster cycle?
            # For now, consistent pulse.
            time.sleep(2)

    # --- Node 5: Growth Engine (The Evolution) ---
    def growth_vibration(field):
        from Core.Divine.self_maintenance_hub import get_maintenance_hub
        from Core.Divine.autonomy_protocol import AutonomyProtocol
        
        hub = get_maintenance_hub()
        protocol = AutonomyProtocol()
        
        # Initial scan
        field["growth_status"] = "Initializing..."
        
        while field["is_alive"]:
            # 1. Diagnose System
            diagnosis = hub.diagnose()
            field["health_score"] = diagnosis.health_score
            
            # 2. Check for Issues
            if diagnosis.health_score < 1.0:
                issues = hub.identify_issues()
                if issues:
                    # 3. Propose Fix
                    # Pick the first critical issue for now
                    # (In a real scenario, we'd prioritize)
                    target_file = issues[0].file_path # Hypothtical structure
                    # Just passing the file path to propose generic fix
                    # Note: identify_issues returns generic objects, we assume they have path.
                    # As a safe fallback, we use the bottleneck path from diagnosis if available.
                    
                    target_path = None
                    if diagnosis.bottlenecks:
                        # Format "filename (metric)"
                        target_path = diagnosis.bottlenecks[0].split(" ")[0]
                        
                    if target_path:
                        plan = hub.propose_fix(target_path)
                        
                        if plan:
                            # 4. Seek Consent (The Bridge)
                            consent_report = protocol.check_consent(plan, field)
                            
                            if consent_report["consent"]:
                                field["thought_log"].append(f"[GROWTH] Mutating {os.path.basename(target_path)}... ({consent_report['reason']})")
                                success = hub.execute_with_consent(plan, consent=True)
                                if success:
                                    field["thought_log"].append("[GROWTH] Mutation Successful.")
                            else:
                                field["thought_log"].append(f"[GROWTH] Mutation Denied: {consent_report['reason']}")

            # Evolution Cycle (Slow)
            field["growth_status"] = "Gestating"
            time.sleep(10) # Check every 10 seconds for now (usually minutes/hours)

    # Initialize Field Defaults
    # Sovereign Level by default (per User Request)
    
    orchestra.add_node("Mind", mind_vibration)
    orchestra.add_node("Heart", heart_vibration)
    orchestra.add_node("Spirit", dream_vibration)
    orchestra.add_node("Growth", growth_vibration) # [Phase 44: Autonomy]
    orchestra.add_node("HUD", hud_vibration)
    
    orchestra.keep_alive()


if __name__ == "__main__":
    main()
