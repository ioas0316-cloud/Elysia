
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional

# Forward declaration for type hinting
class World:
    pass

@dataclass
class Move:
    """Represents a single move (초식) within a martial art style."""
    name: str
    move_type: str  # 'attack', 'defense', 'counter'
    description: str
    apply_effect: Callable[[World, int, int, 'np.ndarray'], float]

    # --- Causal Conditions ---
    min_stats: Dict[str, float] = field(default_factory=dict)
    mp_cost: float = 0.0

@dataclass
class MartialStyle:
    """Represents a martial art style (검법), a collection of moves."""
    name: str
    principle: str  # 'spear', 'net', 'mountain'
    description: str
    moves: List[Move] = field(default_factory=list)

# A registry for all martial styles available in the world.
MARTIAL_STYLES: Dict[str, MartialStyle] = {}

def register_style(style: MartialStyle):
    """Registers a martial art style to make it available in the world."""
    if style.name in MARTIAL_STYLES:
        raise ValueError(f"Style with name '{style.name}' is already registered.")
    MARTIAL_STYLES[style.name] = style

# --- Effect Functions ---
# These functions contain the actual logic of what a move does.

def _effect_basic_attack(world: World, actor_idx: int, target_idx: int, hp_deltas: 'np.ndarray') -> float:
    """A standard attack dealing 1.0x damage."""
    return 1.0

def _effect_piercing_light(world: World, actor_idx: int, target_idx: int, hp_deltas: 'np.ndarray') -> float:
    """A powerful ultimate move that deals 3.0x damage."""
    world.logger.info(f"'{world.cell_ids[actor_idx]}' channels their focus into a [Piercing Light]!")
    return 3.0

# --- Style Definitions ---

# 1. 섬광검법 (Flash Sword Style)
flash_sword_style = MartialStyle(
    name="섬광검법(閃光劍法)",
    principle='spear',
    description="민첩성을 극대화하여, 눈으로 따라갈 수 없는 속도로 적을 베는 검법."
)

flash_sword_style.moves.extend([
    Move(
        name="섬광일섬(閃光一閃)",
        move_type='attack',
        description="가장 기본적인 찌르기 초식.",
        min_stats={'agility': 10},
        mp_cost=0,
        apply_effect=_effect_basic_attack
    ),
    Move(
        name="꿰뚫는 빛(貫通光)",
        move_type='attack',
        description="모든 것을 꿰뚫는 필살기.",
        min_stats={'agility': 40, 'wisdom': 20},
        mp_cost=30,
        apply_effect=_effect_piercing_light
    )
])

# --- Register Styles ---
register_style(flash_sword_style)
