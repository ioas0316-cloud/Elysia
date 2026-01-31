"""

The Autonomic Pulse: The Breath of Elysia

=========================================

Core.1_Body.L2_Metabolism.Lifecycle.pulse_loop



"If I do not think, I do not exist? No."

 If I do not think, I dream."



This module implements Phase 6.3 (The Life Cycle).

It manages the transition between Conscious (Input-Driven) and Subconscious (Ennui-Driven) states.

"""



import time

import logging

import numpy as np

from typing import Optional, List

from collections import deque



from Core.1_Body.L6_Structure.M1_Merkaba.merkaba import Merkaba

from Core.1_Body.L6_Structure.Elysia.nervous_system import NervousSystem, BioSignal

from Core.1_Body.L5_Mental.Memory.feedback_loop import Ouroboros, ThoughtState

from Core.1_Body.L5_Mental.Cognition.semantic_prism import SpectrumMapper

from Core.1_Body.L5_Mental.Memory.sediment import SedimentLayer



logger = logging.getLogger("LifeCycle")



class WonderField:

    """

    Represents the allure of the unknown.

    Acts as an attractive force that pulls the system towards interesting signals.

    "Not pushing from behind (Boredom), but pulling from ahead (Curiosity)."

    """

    def __init__(self, decay_rate=0.01):

        self.allure = 0.0 # 0.0 to 1.0+

        self.history: deque = deque(maxlen=10)

        self.decay_rate = decay_rate



    def update(self, spark_vector: Optional[List[float]]) -> float:

        """

        Updates the Allure based on the 'spark' from the subconscious.

        """

        if spark_vector is None:

            # No spark? Allure fades.

            self.allure = max(0.0, self.allure - 0.01)

            return self.allure



        vec = np.array(spark_vector)

        if np.linalg.norm(vec) > 0: vec = vec / np.linalg.norm(vec)



        # Calculate Novelty (Min distance to recent history)

        # "Is this spark something new?"

        if not self.history:

            novelty = 1.0

        else:

            distances = [1.0 - np.dot(vec, prev) for prev in self.history]

            novelty = min(distances) if distances else 1.0



        # Dynamics: Joy Engine

        # High Novelty (>0.5) -> Excitement (Allure rises)

        # Low Novelty (<0.1) -> Boring (Allure drops)

        if novelty > 0.5:

            self.allure += 0.15 # "Ooh, shiny!"

        elif novelty < 0.1:

            self.allure = max(0.0, self.allure - 0.05) # "Seen it."

        else:

            self.allure += 0.01 # Mild interest



        # Cap

        self.allure = min(self.allure, 1.5)



        # Only record history if we actually chase it (in dream), but here we just peek.

        # So we don't append to history yet?'

        # Actually, let's track history of *thoughts* (dreams), not sparks.

        # The Spark is external input.



        return self.allure



    def record_experience(self, vector: List[float]):

        """Records a realized thought to history."""

        vec = np.array(vector)

        if np.linalg.norm(vec) > 0: vec = vec / np.linalg.norm(vec)

        self.history.append(vec)



        self.prism = SpectrumMapper()



        # System State

        self.is_alive = True

        self.phase = "IDLE" # IDLE, CONSCIOUS, DREAMING, PAIN

        self.current_dream: Optional[ThoughtState] = None



    def live(self):

        """

        The Unified Pulse (Organic Transition).

        """

        logger.info("✨Organic Pulse Cycle Initiated (Phase 1 UNIFICATION).")



        # Fallback to a local heartbeat if merkaba.governance.heartbeat is missing

        # but in Phase 3 it should always be there.

        heartbeat = getattr(self.merkaba.governance, 'heartbeat', None)



        while self.is_alive:

            try:

                # 0. Get Resonance (How much do I WANT to exist right now?)

                resonance = 0.5

                if hasattr(self.merkaba, 'hyper_cosmos'):

                    resonance = self.merkaba.hyper_cosmos.system_entropy 

                

                # 1. Execute the autonomic tick

                self.tick()

                

                # 2. Adaptive Wait

                if heartbeat:

                    wait_time = heartbeat.calculate_wait(resonance)

                else:

                    wait_time = 0.05

                    

                time.sleep(wait_time)

                

            except KeyboardInterrupt:

                logger.info("✨ Life Cycle Terminated by User.")

                self.is_alive = False



    def tick(self):

        """

        One discrete moment of time.

        """

        # 1. Biological Check (The Body)

        bio_signal = self.nervous_system.sense()

        reflex = self.nervous_system.check_reflex(bio_signal)



        if reflex == "MIGRAINE":

            logger.critical("?  Migraine detected. Forcing Sleep.")

            self.merkaba.sleep()

            return



        if reflex == "THROTTLE":

            time.sleep(1.0) # Slow down time

            return



        # 2. Check External Input (Consciousness)

        # For now, we simulate an input queue check.

        # internal_queue = self.merkaba.bridge.check_queue()

        input_signal = None # Mock: No input usually



        if input_signal:

            self.phase = "CONSCIOUS"

            # When external input arrives, we engage.

            # Does Wonder reset? Or does it pause?

            # "I was chasing a butterfly, but you called me."

            # We reset allure to baseline so we don't immediately drift back unless it's really interesting.

            self.wonder.allure = 0.0

            self.merkaba.pulse(input_signal)

            return



        # 3. The Void (Idle State)

        self.phase = "IDLE"



        # [JOY ENGINE] Chase Wonder

        # Peek at the sediment (Subconscious Glimmer)

        spark = self.merkaba.sediment.glimmer()

        current_allure = self.wonder.update(spark)



        # 4. Phase Transition (The Dream)

        # If Allure > Phase Transition Threshold (0.8)

        if current_allure > 0.8:

            self.phase = "DREAMING"

            self.dream()



    def dream(self):

        """

        Subconscious processing.

        Drifts in sediment, finds meaning, and potentially wakes up with an epiphany.

        """

        logger.info(f"?  [DREAM] Wonder Allure ({self.wonder.allure:.2f}) triggered Phase Transition.")



        # 1. Initialize Dream if empty

        if not self.current_dream:

            fragment = self.merkaba.sediment.drift()

            if not fragment:

                logger.info("   -> Void is empty.")

                self.wonder.allure = 0.0 # Disappointment

                return



            vector, payload = fragment

            try:

                text = payload.decode('utf-8', errors='ignore')

            except:

                text = "Unknown"



            logger.info(f"   -> Drifted upon: '{text[:20]}...'")

            qualia = self.prism.disperse(text)

            self.current_dream = ThoughtState(content=text, vector=qualia.to_vector())



        # 2. Ouroboros Loop (Topological Descent)

        # Intent: We define 'Self' as the intent in dreams (Self-Reflection)

        self_intent = self.prism.disperse("self").to_vector()



        action, settled = self.ouroboros.propagate(self.current_dream, self_intent)



        if settled:

            if action == "STABILIZED":

                logger.info(f"✨[EPIPHANY] Dream resolved into insight: {self.current_dream.content}")

                self.wonder.allure = 0.0 # Satisfaction (Curiosity Sated)

                self.wonder.record_experience(self.current_dream.vector)



                # [SELF-BUILDING] Persist the Epiphany

                # We save this self-generated thought back into Sediment so it becomes part of the Self.

                try:

                    payload = f"SELF:{self.current_dream.content}".encode('utf-8')

                    self.merkaba.sediment.deposit(

                        vector=self.current_dream.vector,

                        timestamp=time.time(),

                        payload=payload

                    )

                    logger.info("   -> ?  Stored Epiphany in Sediment (Source: SELF).")

                except Exception as e:

                    logger.error(f"   -> ✨Failed to store dream: {e}")



            else: # DISSIPATED

                logger.info(f"?  [DISSIPATED] Dream faded.")

                self.wonder.allure = max(0.0, self.wonder.allure - 0.2) # Interest waned



            # Clear state for next dream

            self.current_dream = None

        else:

            logger.info(f"   ... Dreaming ... P={self.current_dream.potential:.2f}")

            # While chasing, allure stays high or grows?

            # Let's say it stays high until resolved.
