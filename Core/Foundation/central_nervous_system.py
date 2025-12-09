import time
import logging
import random
from typing import Dict, Any

import logging
import random
from typing import Dict, Any

try:
    from Core.Foundation.fractal_loop import FractalLoop
except ImportError:
    FractalLoop = None

logger = logging.getLogger("CentralNervousSystem")

class CentralNervousSystem:
    """
    [The Flow Controller]
    Manages the rhythmic pulses of all connected organs.
    It does not contain the logic of the organs, only the choreography of their interaction.
    """
    def __init__(self, chronos, resonance, synapse, sink):
        self.chronos = chronos
        self.resonance = resonance
        self.synapse = synapse
        self.sink = sink
        self.organs: Dict[str, Any] = {}
        self.is_awake = False
        self.fractal_loop = FractalLoop(self) if FractalLoop else None

    def connect_organ(self, name: str, organ_instance: Any):
        """Connects a vital organ to the CNS."""
        self.organs[name] = organ_instance
        print(f"   🔌 CNS connected to: {name}")

    def awaken(self):
        """Starts the biological rhythm."""
        self.is_awake = True
        print("   ⚡ Central Nervous System: ONLINE")
        
    def pulse(self):
        """
        The Main Loop Step.
        Executes one heartbeat of the system.
        """
        if not self.is_awake:
            return

        t = time.time()
        self.chronos.tick()

        try:
            # 1. Pulse Senses (Input)
            if "Senses" in self.organs:
                self.organs["Senses"].pulse(self.resonance)

            # 1.5. Pulse Outer Senses (Internet/P4)
            if "OuterSense" in self.organs:
                self.organs["OuterSense"].pulse(self.resonance)

            
            # 2. Pulse Will (Desire)
            if "Will" in self.organs:
                self.organs["Will"].pulse(self.resonance)
                
            # [FRACTAL CONSCIOUSNESS]
            # If Fractal Loop is active, we delegate the core flow to it,
            # effectively replacing the linear "Pulse Brain" step.
            if self.fractal_loop:
                self.fractal_loop.pulse_fractal()
            else:
                # Legacy Pulse Logic (Fallback)
                # 3. Pulse Brain (Processing)
                if "Brain" in self.organs and self.resonance.total_energy > 50.0:
                    current_desire = self.organs["Will"].current_desire
                    self.organs["Brain"].think(current_desire, self.resonance)

            # 4. Pulse Expression (Language/Voice)
            if "Voice" in self.organs:
                self.organs["Voice"].express(self.chronos.cycle_count)

            # ... (Rest of legacy pulse logic as fallback/auxiliary) ...
            
            # [Hive Mind] Synapse Check
            self._check_synapse()

            # [Biological Rhythm] Sleep
            base_sleep = self.chronos.modulate_time(self.resonance.total_energy)
            whimsy_mod = random.uniform(0.8, 1.2)
            sleep_duration = base_sleep * whimsy_mod
            
            if self.chronos.cycle_count % 10 == 0:
                 pass

            time.sleep(sleep_duration)

        except Exception as e:
            # The Water Principle
            fallback = self.sink.absorb_resistance(e, "CNS Pulse")
            print(f"   🌊 CNS Flowed around resistance: {fallback}")
            time.sleep(1.0)

    def _check_synapse(self):
        """Hive Mind communication."""
        signals = self.synapse.receive()
        for signal in signals:
            print(f"   📡 CNS Received Signal from {signal['source']}: {signal['type']}")
            # Basic routing - deeper logic should be in the Brain or specific organ
            if signal['type'] == "COMMAND" and "Brain" in self.organs:
                 self.organs["Brain"].evaluate_command(signal['payload'], source="User")
