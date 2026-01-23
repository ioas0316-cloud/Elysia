from typing import List, Dict, Tuple

class InsightSynthesizer:
    """
    Synthesizes lists of static and dynamic facts from the LogicalReasoner
    into a coherent, natural language insight.
    """

    def synthesize(self, facts: List[str]) -> str:
        """
        Takes a list of facts and synthesizes them into a single, insightful paragraph.

        Args:
            facts: A list of strings, where each string is a fact.
                   Facts can be prefixed with "[  ]" or be part of a simulation result block.

        Returns:
            A natural language string representing the synthesized insight.
        """
        if not facts:
            return "                            .                   ."

        static_facts, dynamic_facts, sim_header = self._classify_facts(facts)

        if static_facts and dynamic_facts:
            # Both static knowledge and dynamic simulation results are present
            return self._synthesize_combined_insight(static_facts, dynamic_facts, sim_header)
        elif static_facts:
            # Only static knowledge is present
            return self._synthesize_static_insight(static_facts)
        elif dynamic_facts:
            # Only dynamic simulation results are present
            return self._synthesize_dynamic_insight(dynamic_facts, sim_header)
        else:
            # Handle cases where classification fails or facts are empty
            return "                        .             ."

    def _classify_facts(self, facts: List[str]) -> Tuple[List[str], List[str], str]:
        """Helper to classify facts into static, dynamic, and the simulation header."""
        static_facts = []
        dynamic_facts = []
        sim_header = ""

        is_dynamic_block = False
        for fact in facts:
            if "         " in fact:
                is_dynamic_block = True
                sim_header = fact.replace(":", "").strip()
                continue

            if is_dynamic_block and "            " in fact:
                # Clean up the dynamic fact for synthesis
                clean_fact = fact.strip().replace("  - ", "")
                dynamic_facts.append(clean_fact)
            elif "[  ]" in fact:
                # Clean up the static fact
                clean_fact = fact.replace("[  ]", "").strip()
                static_facts.append(clean_fact)

        return static_facts, dynamic_facts, sim_header

    def _synthesize_combined_insight(self, static_facts: List[str], dynamic_facts: List[str], sim_header: str) -> str:
        """Synthesizes an insight when both static and dynamic facts are available."""
        static_summary = " ".join(static_facts)
        dynamic_summary = ", ".join(dynamic_facts)

        return (
            f"               .                {static_summary}  , "
            f"    {sim_header}           {dynamic_summary}            . "
            "                     ,                          ."
        )

    def _synthesize_static_insight(self, static_facts: List[str]) -> str:
        """Synthesizes an insight based only on static knowledge."""
        if len(static_facts) == 1:
            return f"       , {static_facts[0]}."
        else:
            summary = "     ".join(static_facts)
            return f"                            .      , {summary}."

    def _synthesize_dynamic_insight(self, dynamic_facts: List[str], sim_header: str) -> str:
        """Synthesizes an insight based only on dynamic simulation."""
        dynamic_summary = ", ".join(dynamic_facts)
        return (
            f"           , {sim_header}                   "
            f"{dynamic_summary}               .                                           ."
        )

if __name__ == '__main__':
    # --- Test Cases ---
    synthesizer = InsightSynthesizer()

    # Case 1: Combined facts
    combined_facts = [
        "[  ] '  ' ( ) '     ' ( )           .",
        "'  '( )           ,                      :",
        "  - '     '              (   : 0.41).",
        "  - '     '              (   : 0.2)."
    ]
    print("--- Combined Insight ---")
    print(synthesizer.synthesize(combined_facts))

    # Case 2: Static facts only
    static_only = [
        "[  ] '     ' ( ) '  '         ."
    ]
    print("\n--- Static Insight ---")
    print(synthesizer.synthesize(static_only))

    # Case 3: Dynamic facts only
    dynamic_only = [
        "'  '( )           ,                      :",
        "  - '  '              (   : 0.8).",
        "  - '  '              (   : 0.6)."
    ]
    print("\n--- Dynamic Insight ---")
    print(synthesizer.synthesize(dynamic_only))

    # Case 4: No facts
    print("\n--- No Facts ---")
    print(synthesizer.synthesize([]))