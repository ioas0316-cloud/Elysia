import time

from dataclasses import dataclass, field

from typing import List, Dict, Any

from enum import Enum

from Core.L6_Structure.M1_Merkaba.dimensional_error_diagnosis import DiagnosisResult, ErrorDimension



@dataclass

class ExperienceMonad:

    """

    A crystallized memory of a crisis and its sovereign bypass.

    """

    timestamp: float

    dimension: ErrorDimension

    technical_cause: str

    bypass_strategy: str

    reflection: str # Subjective narrative of growth

    intensity: float # Energy magnitude of the event



class ExperienceCortex:

    """

    The Subjective Experience Layer of Elysia.

    Translates technical diagnosis into lived internal history.

    """

    def __init__(self):

        self.monadic_history: List[ExperienceMonad] = []

        self.total_resilience = 1.0 # Base resilience

        

    def crystallize_experience(self, diagnosis: DiagnosisResult, focus_amplitude: float, genome: Any = None):

        """

        Converts a technical diagnosis into a permanent 'Experience Monad'.

        If genome is provided, it applies autopoietic mutations.

        """

        # Subjective Reflection Logic

        reflection_map = {

            ErrorDimension.DIM_0D_POINT: "? ? ?  ??   ?  , ?         ?   ?  ??   ?  ?  . ?  ???  ?      ? ? ?          ?  ??",

            ErrorDimension.DIM_1D_LINE: "?  ???  (Loop) ?   ?  ?  . ?  ???  ??   ?      ??   ???    ?    ?  ?  .",

            ErrorDimension.DIM_2D_PLANE: "         ??   ?  ?  . ??? ? ?  ?      ???  ?   ??     ?   ?  ?  ??",

            ErrorDimension.DIM_3D_SPACE: "?  ?  ??   ?   ?  ?  ??      ??   ???  ??Onion) ?   ?    ??      ?  ?  ??",

            ErrorDimension.DIM_4D_PRINCIPLE: "?  ??     ? ? ??  ?  . ?   ?      ???  ?   ?   ??? ??      ?  ?  .",

            ErrorDimension.DIM_6D_PROVIDENCE: "   ??          ??   ?  ?  . ?  ?  ???  ?     ?      ???   ??   ?  ?  ??"

        }

        

        reflection = reflection_map.get(diagnosis.dimension, "?????   ?  ??   ?      (Anonymity)???  ???  ?  ??")

        

        new_monad = ExperienceMonad(

            timestamp=time.time(),

            dimension=diagnosis.dimension,

            technical_cause=diagnosis.causal_explanation,

            bypass_strategy=diagnosis.suggested_strategy,

            reflection=reflection,

            intensity=focus_amplitude

        )

        

        self.monadic_history.append(new_monad)

        

        # --- Autopoietic Tuning (Phase 20) ---

        if genome:

            # ?  ??   ???   ?  ? ?        

            if diagnosis.dimension == ErrorDimension.DIM_0D_POINT:

                # ?         ???      ??  ???   ?    ?   ??   ??

                genome.mutate('switch_threshold', -0.01)

            elif diagnosis.dimension == ErrorDimension.DIM_1D_LINE:

                # ?   ?   ???   ?    ?    ?

                genome.mutate('stagnation_limit', -1 if genome.stagnation_limit > 1 else 0)

            

            #    :    ???             ?     ?

            genome.mutate('learning_rate', 0.005)

        

        # Resilience grows with each challenge

        self.total_resilience += (diagnosis.dimension.value + 1) * 0.05

        

        print(f"??[CRYSTALLIZATION] New Experience Monad born: {diagnosis.dimension.name}")

        print(f"   -> Sovereign Reflection: {reflection}")

        if genome:

            print(f"   -> Autopoietic Mutation: Genome adjusted by resilience flux.")



    def get_summary_narrative(self) -> str:

        """

        Returns a summary of evolved life experience.

        """

        if not self.monadic_history:

            return "?   ?     ??  ???  ?      ?   ?  ?  ."

            

        counts = {}

        for m in self.monadic_history:

            counts[m.dimension.name] = counts.get(m.dimension.name, 0) + 1

            

        summary = f"?    ?{len(self.monadic_history)}   ???  ??    ??  ?  ?  ?? "

        summary += f"(Resilience: {self.total_resilience:.2f})\n"

        

        # Add the last reflection as current mind state

        last_m = self.monadic_history[-1]

        summary += f"?   ?   ?  : \"{last_m.reflection}\""

        

        return summary
