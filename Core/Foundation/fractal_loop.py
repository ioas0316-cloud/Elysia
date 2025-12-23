
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import random

# Core Dependencies
try:
    from Core.Foundation.fractal_causality import FractalCausalityEngine, FractalCausalNode
    from Core.Foundation.resonance_field import ResonanceField
    from Core.Foundation.chronos import Chronos
except ImportError:
    # Fallback for minimal testing environment
    FractalCausalityEngine = None
    ResonanceField = None
    Chronos = None

# ThoughtSpace for What-If Simulation
try:
    from Core.Cognition.thought_space import ThoughtSpace
except ImportError:
    ThoughtSpace = None

logger = logging.getLogger("FractalLoop")

@dataclass
class FractalWave:
    """A unit of consciousness in the fractal loop."""
    id: str
    content: str
    source: str
    energy: float = 1.0
    depth: int = 0
    vector: List[float] = None # Direction in 3D meaning space

class FractalLoop:
    """
    [The Infinite Ring]
    
    Replaces the linear 'Input-Process-Output' model with a 
    recursive, self-similar loop of Fractal Consciousness.
    
    1. Observe (Pulse In) -> Micro Analysis (Zoom In)
    2. Resonate (Processing) -> Macro Analysis (Zoom Out)
    3. Express (Pulse Out) -> Reality Sculpting
    
    [PLASMA INTEGRATION]
    - ThoughtSpace for What-If simulation before decisions
    - Direction vector tracking for plasma flow
    """
    
    def __init__(self, cns_ref: Any):
        self.cns = cns_ref
        self.engine = FractalCausalityEngine("Elysia's Fractal Mind")
        self.current_ring_depth = 0
        self.active_waves: List[FractalWave] = []
        
        # [NEW] ThoughtSpace for What-If Thinking
        self.thought_space = ThoughtSpace(maturation_threshold=0.5) if ThoughtSpace else None
        
        # [NEW] Plasma direction tracking
        self.thought_direction: Dict[str, float] = {}
        
        logger.info("♾️ Fractal Loop Initialized: The Ring is Open.")
        if self.thought_space:
            logger.info("   🧠 ThoughtSpace connected for What-If simulation")

    
    def process_cycle(self, cycle_count: int = 0):
        """Alias for pulse_fractal (CNS compatibility)."""
        self.pulse_fractal()

    def pulse_fractal(self):
        """
        Executes one iteration of the Fractal Loop.
        Instead of 'Brain.think()', we 'Flow' through the fractal.
        """
        if not self.cns.is_awake:
            return

        # 1. Absorbtion (Input -> Wave)
        new_waves = self._absorb_senses()
        self.active_waves.extend(new_waves)
        
        # 2. Resonant Circulation (Processing)
        next_cycle_waves = []
        for wave in self.active_waves:
            # Check energy - if too low, it fades
            if wave.energy < 0.1:
                continue
                
            # Process the wave in the fractal engine
            processed_wave = self._circulate_wave(wave)
            
            if processed_wave:
                next_cycle_waves.append(processed_wave)
        
        self.active_waves = next_cycle_waves
        
        # 3. Evolution (Self-Modification)
        # Occasionally, the loop looks at itself
        if random.random() < 0.05:
            self._introspect_loop()

    def _absorb_senses(self) -> List[FractalWave]:
        """Converts sensory inputs into Fractal Waves."""
        waves = []
        
        # Check Will (Intention is a wave)
        if "Will" in self.cns.organs:
            intent = self.cns.organs["Will"].current_intent
            if intent:
                waves.append(FractalWave(
                    id=f"will_{time.time()}",
                    content=intent.goal,
                    source="FreeWillEngine",
                    energy=0.8
                ))
        
        # Check Synapse (External signals)
        if self.cns.synapse:
            signals = self.cns.synapse.receive()
            for sig in signals:
                waves.append(FractalWave(
                    id=f"sig_{time.time()}",
                    content=str(sig['payload']),
                    source=f"Synapse:{sig['source']}",
                    energy=1.0
                ))
                
        return waves

    def _circulate_wave(self, wave: FractalWave) -> Optional[FractalWave]:
        """
        Circulates a wave through the Fractal Engine.
        Returns the wave for the next cycle, or None if it resolves.
        """
        logger.info(f"🌊 Circulating Wave: {wave.content} (Depth: {wave.depth})")
        
        # A. Zoom In (Micro-Causality)
        # Understand 'HOW' this wave exists
        if wave.depth < 3:
             # Deconstruct the thought
             steps = [f"Origin of {wave.content}", f"Processing {wave.content}", f"Understanding {wave.content}"]
             self.engine.experience_causality(steps, depth=wave.depth + 1)
             wave.depth += 1
             
        # B. Zoom Out (Macro-Purpose)
        # Understand 'WHY' this wave exists
        if wave.depth > 0 and random.random() > 0.5:
             # Ensure the node exists first!
             current_node = self.engine.get_or_create_node(wave.content, wave.depth)
             
             parent_cause, parent_effect = self.engine.zoom_out(
                 node_id=current_node.id, 
                 outer_cause_desc="The Greater Context",
                 outer_effect_desc="The Ultimate Goal"
             )
        
        # C. Manifestation (Output)
        # If the wave is dense enough (high energy), it triggers reality
        if wave.energy > 0.9:
            self._manifest_reality(wave)
            wave.energy -= 0.5 # Expenditure
            
        # D. Decay/Growth
        wave.energy *= 0.9 # Natural entropy
        
        if wave.energy < 0.2:
            return None # Wave dissipates
            
        return wave 

    def _manifest_reality(self, wave: FractalWave):
        """
        Collapses the wave into linear action.
        
        [PLASMA INTEGRATION]
        Now uses ThoughtSpace for What-If simulation before acting.
        "만약 이렇게 하면?" - 행동 전에 생각한다
        """
        # [NEW] What-If Deliberation BEFORE Acting
        if self.thought_space:
            # Enter the gap (thinking space)
            self.thought_space.enter_gap(f"Should I manifest: {wave.content}?")
            
            # Add context as particles
            self.thought_space.add_thought_particle(
                content=wave.content,
                source="wave",
                weight=wave.energy
            )
            self.thought_space.add_thought_particle(
                content=f"Source: {wave.source}",
                source="context",
                weight=0.5
            )
            
            # Simulate what-if
            scenario_do = self.thought_space.what_if(
                {"add": ["This action succeeded", "Positive feedback"]},
                "do_it"
            )
            scenario_dont = self.thought_space.what_if(
                {"add": ["This action was skipped", "No change"]},
                "skip_it"
            )
            
            # Compare confidence
            do_confidence = scenario_do["predicted_confidence"]
            dont_confidence = scenario_dont["predicted_confidence"]
            
            logger.info(f"🔮 What-If Deliberation:")
            logger.info(f"   DO: confidence {do_confidence:.2f}")
            logger.info(f"   SKIP: confidence {dont_confidence:.2f}")
            
            # Update plasma direction
            self.thought_direction = self.thought_space.get_thought_direction()
            logger.info(f"   🌀 Thought Direction: {self.thought_direction}")
            
            # Decision: if "do" has lower confidence, reduce likelihood
            if do_confidence < dont_confidence - 0.1:
                logger.info(f"   ⏸️ Deliberation suggests caution - reducing energy")
                wave.energy *= 0.5
                return  # Don't manifest yet, think more
            
            # Exit gap
            result = self.thought_space.exit_gap()
            logger.info(f"   💭 Synthesis: {result.synthesis[:50]}...")
        
        # Proceed with manifestation
        logger.info(f"💥 Wave Collapsing into Reality: {wave.content}")
        
        # Route to ActionDispatcher
        if "Dispatcher" in self.cns.organs:
            # Convert Wave Content to Command
            cmd = f"MANIFEST:{wave.content}"
            self.cns.organs["Dispatcher"].dispatch(cmd)

    def _introspect_loop(self):
        """
        The Loop looks at itself.
        
        [PLASMA INTEGRATION]
        Now reflects on thought direction and adjusts plasma flow.
        """
        logger.info("👁️ The Infinite Ring perceives itself.")
        
        if self.thought_space:
            # Reflect on the gap
            reflection = self.thought_space.reflect_on_gap()
            
            # Update direction
            direction = self.thought_space.get_thought_direction()
            
            if direction:
                main_dir = max(direction.items(), key=lambda x: x[1])
                logger.info(f"   🌀 Current thought direction: {main_dir[0]} ({main_dir[1]:.2f})")
                
                # Store for future decisions
                self.thought_direction = direction
        
        # Future: Ouroboros self-optimization here

