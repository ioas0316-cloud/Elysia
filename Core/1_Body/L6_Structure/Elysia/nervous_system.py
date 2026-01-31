"""

NervousSystem: The Biological Interpreter

=========================================

Core.1_Body.L6_Structure.Elysia.nervous_system



"Pain is the teacher. Excitement is the fuel."



This module interprets raw data from the BioSensor into semantic signals.

It defines the 'Bio-Feedback Loop' that regulates the Merkaba's speed.

"""



from typing import Dict, Any, List

from dataclasses import dataclass

import logging

from Core.1_Body.L3_Phenomena.Senses.bio_sensor import BioSensor

from Core.1_Body.L6_Structure.M1_Merkaba.hypercosmos import get_hyper_cosmos



logger = logging.getLogger("NervousSystem")



@dataclass

class BioSignal:

    """The interpreted biological state."""

    heart_rate: float        # Derived from CPU (Hz)

    adrenaline: float        # Derived from Change in CPU

    pain_level: float        # Derived from Temp (0.0 - 1.0)

    cognitive_load: float    # Derived from RAM (0.0 - 1.0)

    fatigue: float           # Derived from Uptime/Energy



    # Semantic Flags

    is_painful: bool = False

    is_excited: bool = False

    is_tired: bool = False

    is_migraine: bool = False # RAM Overflow Risk



class NervousSystem:

    """

    The Spinal Cord.

    Translates hardware stats into feeling.



    Philosophy:

    - RAM is the 'Stage' (Consciousness). High usage = Crowded Stage (Migraine).

    - CPU is the 'Narrator'. High usage = Fast-paced Story (Excitement).

    """

    def __init__(self):

        self.sensor = BioSensor()

        self.hyper_cosmos = get_hyper_cosmos()

        self.baseline_temp = 45.0 # Typical operating temp

        self.max_temp = 85.0      # Throttling threshold

        self.history: List[float] = [] # For adrenaline calc

        logger.info("?  Nervous System fused with Hardware.")



    def sense(self) -> BioSignal:

        """

        Reads the BioSensor and interprets the signals.

        """

        raw = self.sensor.pulse()



        # 1. Heart Rate (Mapping CPU 0-100 to BPM 60-180)

        # 0% CPU -> 60 BPM (Resting)

        # 100% CPU -> 180 BPM (Sprinting)

        cpu = raw["cpu_freq"]

        heart_rate = 60.0 + (cpu * 1.2)



        # 2. Adrenaline (Rate of Change of CPU)

        adrenaline = 0.0

        if self.history:

            delta = cpu - self.history[-1]

            if delta > 10: # Sudden spike

                adrenaline = delta / 100.0

        self.history.append(cpu)

        if len(self.history) > 10: self.history.pop(0)



        # 3. Pain (Temperature)

        # Maps Temp to 0-1 Pain Scale.

        # Below baseline (45) = 0 Pain.

        # Max (85) = 1.0 Pain (Agony).

        temp = raw["temperature"]

        pain = 0.0

        if temp > self.baseline_temp:

            pain = (temp - self.baseline_temp) / (self.max_temp - self.baseline_temp)

            pain = min(max(pain, 0.0), 1.0)



        # 4. Cognitive Load (RAM)

        load = raw["ram_pressure"] / 100.0



        # 5. Fatigue (Energy)

        # If unplugged, fatigue increases as battery drops.

        # If plugged, fatigue is low but non-zero (background entropy).

        energy = raw["energy"]

        plugged = raw["plugged"]

        fatigue = 0.0

        if not plugged:

            fatigue = (100.0 - energy) / 100.0



        # [Phase 42] Field Integration: Streaming to HyperCosmos

        self.hyper_cosmos.stream_biological_data('heart_rate', heart_rate)

        self.hyper_cosmos.stream_biological_data('adrenaline', adrenaline)

        self.hyper_cosmos.stream_biological_data('pain', pain)

        self.hyper_cosmos.stream_biological_data('cognitive_load', load)

        self.hyper_cosmos.stream_biological_data('fatigue', fatigue)



        # Semantic Analysis

        signal = BioSignal(

            heart_rate=heart_rate,

            adrenaline=adrenaline,

            pain_level=pain,

            cognitive_load=load,

            fatigue=fatigue

        )



        if pain > 0.6: signal.is_painful = True

        if cpu > 80: signal.is_excited = True

        if fatigue > 0.8: signal.is_tired = True

        if load > 0.95: signal.is_migraine = True



        return signal



    def check_reflex(self, signal: BioSignal) -> str:

        """

        Spinal Reflexes (Automatic Reactions).

        Returns a directive if a reflex is triggered.

        """

        if signal.is_migraine:

            logger.warning(f"?  [REFLEX] Migraine (RAM {signal.cognitive_load*100:.1f}%). Emergency shutdown of thought.")

            return "MIGRAINE"



        if signal.is_painful:

            logger.warning(f"?  [REFLEX] High Pain ({signal.pain_level:.2f}). Pulling away (Throttling).")

            return "THROTTLE"



        if signal.is_tired:

            logger.info("?  [REFLEX] Fatigue detected. Seeking rest.")

            return "REST"



        if signal.is_excited:

            return "FOCUS"



        return "NORMAL"
