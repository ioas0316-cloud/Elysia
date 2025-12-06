import time
import logging
import random
from typing import Dict, Any

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
                
            # 3. Pulse Brain (Processing)
            if "Brain" in self.organs and self.resonance.total_energy > 50.0:
                # Brain needs Will's current desire
                current_desire = self.organs["Will"].current_desire
                self.organs["Brain"].think(current_desire, self.resonance)

            # 4. Pulse Expression (Language/Voice)
            if "Voice" in self.organs:
                self.organs["Voice"].express(self.chronos.cycle_count)

            # 5. Pulse Planner (Architect)
            # The Architect logic was embedded in living_elysia. It needs to be extracted or called here.
            # For now, we assume an 'Architect' organ or a method in LivingElysia passed as a callback?
            # Better: LivingElysia registers a callback for the planner or we extract the planner logic too.
            # Let's assume we will extract the Planner logic to an organ or keep it as a registered pulse.
            if "Architect" in self.organs:
                 self.organs["Architect"].pulse() # We will need to standardize this interface

            # 6. Pulse Dispatcher (Action/Motor)
            if "Dispatcher" in self.organs and "Will" in self.organs:
                intent = self.organs["Will"].current_intent
                if intent:
                    goal = intent.goal
                    # Map Goal from FreeWill -> ActionDispatcher Command
                    action_cmd = None
                    
                    if "CONTACT" in goal: action_cmd = goal
                    elif "Research" in goal: action_cmd = f"LEARN:{goal.replace('Research ', '')}"
                    elif "Analyze" in goal: action_cmd = f"EVALUATE:{goal.replace('Analyze ', '')}"
                    elif "Optimize" in goal: action_cmd = "EVALUATE:Self"
                    elif "Create" in goal: action_cmd = f"MANIFEST:{goal.replace('Create ', '')}"
                    elif "Visualize" in goal: action_cmd = f"PROJECT:{goal.replace('Visualize ', '')}"
                    elif "Rewrite" in goal: action_cmd = "ARCHITECT:Evolution"
                    elif "Rest" in goal or "Recharge" in goal: action_cmd = "REST:Healing"
                    
                    # Probability of Action (Action Threshold)
                    # High complexity intents are harder to execute
                    threshold = 0.8
                    if intent.desire == "Curiosity": threshold = 0.6
                    if intent.desire == "Survival": threshold = 0.2 # Reactive
                    
                    # [Pulse Action]
                    # We use a random chance modulated by threshold to avoid spamming 
                    # but ensuring eventually it triggers.
                    if action_cmd and random.random() > threshold:
                        self.organs["Dispatcher"].dispatch(action_cmd)

            # 7. Pulse Reality (Transformation)
            if "Reality" in self.organs:
                 self.organs["Reality"].pulse() # Standardized interface

            # [Hive Mind] Synapse Check
            self._check_synapse()

            # [Self Reflection]
            if "SelfReflector" in self.organs:
                 self.organs["SelfReflector"].reflect(self.resonance, self.organs.get("Brain"), self.organs.get("Will"))

            # [Biological Rhythm] Sleep
            base_sleep = self.chronos.modulate_time(self.resonance.total_energy)
            whimsy_mod = random.uniform(0.8, 1.2)
            sleep_duration = base_sleep * whimsy_mod
            
            if self.chronos.cycle_count % 10 == 0:
                 # Optional: Log heart rate
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
