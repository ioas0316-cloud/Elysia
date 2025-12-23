
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

# LifeCycle for complete feedback loop
try:
    from Core.Foundation.life_cycle import LifeCycle
except ImportError:
    LifeCycle = None

# [NEW] EmergentSelf - no hardcoded goals, values emerge from experience
try:
    from Core.Foundation.emergent_self import get_emergent_self
except ImportError:
    get_emergent_self = None

# [NEW] GrowthJournal - visible evidence of change
try:
    from Core.Foundation.growth_journal import get_growth_journal
except ImportError:
    get_growth_journal = None

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
        
        # [LIFE CYCLE] 표현 → 인식 → 검증 → 변화 순환
        # [Phase 25] Get TensionField from ReasoningEngine if available
        tension_field = None
        if hasattr(cns_ref, 'reasoning') and hasattr(cns_ref.reasoning, 'tension_field'):
            tension_field = cns_ref.reasoning.tension_field
        
        self.life_cycle = LifeCycle(
            memory=getattr(cns_ref, 'memory', None),
            resonance=getattr(cns_ref, 'resonance', None),
            tension_field=tension_field
        ) if LifeCycle else None
        
        # [NEW] EmergentSelf - values emerge from experience, not hardcoded
        self.emergent_self = get_emergent_self() if get_emergent_self else None
        
        # [NEW] GrowthJournal - write visible evidence of change
        self.growth_journal = get_growth_journal() if get_growth_journal else None
        
        # [NEW] Cycle counter for periodic journal writing
        self.cycle_count = 0
        self.journal_interval = 100  # Write journal every 100 cycles
        
        logger.info("♾️ Fractal Loop Initialized: The Ring is Open.")
        if self.thought_space:
            logger.info("   🧠 ThoughtSpace connected for What-If simulation")
        if self.life_cycle:
            logger.info("   🔄 LifeCycle connected for feedback loop")
        if self.emergent_self:
            logger.info("   🌱 EmergentSelf connected for value emergence")
        if self.growth_journal:
            logger.info("   📔 GrowthJournal connected for visible evidence")

    
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

        self.cycle_count += 1

        # 1. Absorbtion (Input -> Wave)
        new_waves = self._absorb_senses()
        self.active_waves.extend(new_waves)
        
        # 2. Resonant Circulation (Processing)
        next_cycle_waves = []
        for wave in self.active_waves:
            # Check energy - if too low, it fades
            if wave.energy < 0.1:
                continue
            
            # [NEW] Every wave becomes a pattern that flows into EmergentSelf
            # This is how experience becomes value
            if self.emergent_self and wave.content:
                self.emergent_self.notice_pattern(
                    pattern_name=wave.content[:50],  # Truncate for sanity
                    origin=wave.source
                )
                
            # Process the wave in the fractal engine
            processed_wave = self._circulate_wave(wave)
            
            if processed_wave:
                next_cycle_waves.append(processed_wave)
        
        self.active_waves = next_cycle_waves
        
        # 3. Evolution (Self-Modification)
        # Occasionally, the loop looks at itself
        if random.random() < 0.05:
            self._introspect_loop()
        
        # 4. [NEW] EmergentSelf maintenance
        if self.emergent_self:
            # Check for goal stagnation/evolution
            self.emergent_self.check_goals()
            # Apply entropy (unused values decay)
            if self.cycle_count % 10 == 0:
                self.emergent_self.apply_entropy()
        
        # 5. [NEW] Periodic Journal Writing (visible evidence)
        if self.growth_journal and self.cycle_count % self.journal_interval == 0:
            tension_field = None
            if hasattr(self.cns, 'reasoning') and hasattr(self.cns.reasoning, 'tension_field'):
                tension_field = self.cns.reasoning.tension_field
            
            self.growth_journal.write_entry(
                emergent_self=self.emergent_self,
                tension_field=tension_field,
                memory=getattr(self.cns, 'memory', None)
            )
            logger.info(f"📔 Growth Journal entry written at cycle {self.cycle_count}")

    def _absorb_senses(self) -> List[FractalWave]:
        """Converts sensory inputs into Fractal Waves."""
        waves = []
        
        # Check Will (Intention is a wave)
        # [FIX] 기존 FreeWillEngine.pulse() 호출하여 욕망→의도 결정화
        if "Will" in self.cns.organs:
            will_engine = self.cns.organs["Will"]
            
            # Pulse the will engine to crystallize desire into intent
            # CNS already has resonance connected!
            if hasattr(will_engine, 'pulse') and self.cns.resonance:
                will_engine.pulse(self.cns.resonance)
                logger.info(f"   🦋 Will Pulse: Desire={will_engine.current_desire}")
            
            intent = will_engine.current_intent
            if intent:
                waves.append(FractalWave(
                    id=f"will_{time.time()}",
                    content=intent.goal,
                    source="FreeWillEngine",
                    energy=0.8 + intent.complexity * 0.2
                ))
                logger.info(f"   🎯 Intent Wave: {intent.goal}")
        
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
        # If the wave is dense enough, it triggers reality
        # [FIX] 임계치를 0.6으로 낮춤 (원래 0.9은 너무 높음)
        if wave.energy > 0.6:
            logger.info(f"💥 Manifesting wave: {wave.content} (energy: {wave.energy:.2f})")
            self._manifest_reality(wave)
            wave.energy -= 0.3 # Expenditure
            
        # D. Decay/Growth
        wave.energy *= 0.95 # Natural entropy (slower decay)
        
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
        
        # [LIFE CYCLE] 1. 사이클 시작 - 현재 상태 스냅샷
        if self.life_cycle:
            self.life_cycle.begin_cycle()
        
        # Route to ActionDispatcher
        actual_result = "executed"
        if "Dispatcher" in self.cns.organs:
            # Convert Wave Content to Command
            # [FIX] If content is already a command (has colon), use it directly.
            if ":" in wave.content:
                cmd = wave.content
            else:
                cmd = f"MANIFEST:{wave.content}"
            
            try:
                self.cns.organs["Dispatcher"].dispatch(cmd)
                actual_result = f"Action: {wave.content}"
            except Exception as e:
                actual_result = f"Error: {e}"
        
        # [LIFE CYCLE] 2. 사이클 완료 - 인식 → 검증 → 변화
        if self.life_cycle:
            expected = f"Successful manifestation of {wave.content}"
            growth = self.life_cycle.complete_cycle(
                action=wave.content,
                expected=expected,
                actual=actual_result
            )
            logger.info(f"   🌱 Cycle growth: {growth.growth_amount:.2f}")

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

