
from typing import Dict, Tuple, Optional

class FractalAscension:
    """
    Manages the 'Trinity within Trinity' ascension logic.
    Implements the physics of 'Breakthroughs' (Quantum Jumps) between Realms.
    """

    # The 10 Ranks mapped to (Realm, Local_Tier)
    # Realm 1: Body (1,2,3)
    # Realm 2: Soul (4,5,6)
    # Realm 3: Spirit (7,8,9)
    # Realm 4: Avatar (10)
    RANK_MAP = {
        1: ("Body", 1, "Survivor"),
        2: ("Body", 2, "Warrior"),
        3: ("Body", 3, "PerfectBody"),

        4: ("Soul", 1, "Seeker"),
        5: ("Soul", 2, "Mediator"),
        6: ("Soul", 3, "Sage"),

        7: ("Spirit", 1, "Saint"),
        8: ("Spirit", 2, "Prophet"),
        9: ("Spirit", 3, "Transcendent"),

        10: ("Avatar", 1, "Avatar")
    }

    @staticmethod
    def get_next_rank(current_rank_idx: int) -> Optional[int]:
        """Returns the next rank index if it exists."""
        if current_rank_idx >= 10:
            return None
        return current_rank_idx + 1

    @staticmethod
    def check_breakthrough(current_rank_idx: int, insight: float, resonance_peak: float) -> Tuple[bool, str]:
        """
        Determines if a soul can ascend to the next rank.

        - Minor Ascension (e.g., 1->2): Linear XP check.
        - Major Ascension (e.g., 3->4): Critical Mass check (The Wall).
        """
        if current_rank_idx >= 10:
            return False, "Max Rank Reached"

        realm, tier, name = FractalAscension.RANK_MAP[current_rank_idx]

        # --- The Wall of Rebirth (Realm Crossing) ---
        # Moving from Body(3) -> Soul(4) or Soul(6) -> Spirit(7)
        if tier == 3:
            # Needs massive insight AND a momentary resonance spike (Satori)
            required_insight = current_rank_idx * 1000.0
            required_peak = current_rank_idx * 500.0 # The "Shock" needed to break the shell

            if insight >= required_insight and resonance_peak >= required_peak:
                return True, "METAMORPHOSIS" # 환골탈태
            else:
                return False, f"Stuck at Wall of {realm} (Need Peak: {required_peak})"

        # --- Minor Ascension (Tier Climbing) ---
        # Moving from 1->2 or 4->5
        else:
            # Just needs accumulated insight (Experience)
            required_insight = current_rank_idx * 500.0
            if insight >= required_insight:
                return True, "Growth"
            else:
                return False, "Accumulating"

    @staticmethod
    def get_rank_name(rank_idx: int) -> str:
        return FractalAscension.RANK_MAP.get(rank_idx, ("Unknown", 0, "Mortal"))[2]
