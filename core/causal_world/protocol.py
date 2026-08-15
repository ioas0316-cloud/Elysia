"""
Universal QWER Action Protocol & Layered Skill Tree Architecture
================================================================
This module defines the universal, isomorphic action execution protocol shared identically
by all entities in the causal simulation (Players, NPCs, Guards, Merchants, Demon Lords).

Core Philosophy:
1. Universal Action Protocol: Q, W, E, R execution cycle [Input -> Resource Cost -> Friction -> State Transition].
2. Layered Skill Tree: Skills derived from Origin (Species), Bloodline (Family), Guild (Faction), and Job (Class).
3. Conservation & Destruction Law: Full loot / item destruction creates immediate capability & resource Voids (Deficiencies).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import time
import math


class SkillLayer(Enum):
    ORIGIN = "Origin"         # Species/Origin (Human, Demon, Elf, Spirit, etc.)
    BLOODLINE = "Bloodline"   # Family/Heritage (Swordmaster, Merchant Lineage, Shadow Clan)
    GUILD = "Guild"           # Faction/Affiliation (Order of Knights, Merchant Syndicate, Underground)
    JOB = "Job"               # Class/Role (Warrior, Rogue, Mage, Envoy)


class SlotType(Enum):
    Q = "Q"
    W = "W"
    E = "E"
    R = "R"


@dataclass
class Stats:
    STR: float = 10.0  # Strength (Physical Friction / Damage)
    VIT: float = 10.0  # Vitality (HP / Defense / Physical Endurance)
    INT: float = 10.0  # Intelligence (Information Resolution / MP / Spells)
    AGI: float = 10.0  # Agility (Mobility / Speed / Evasion)
    SPR: float = 10.0  # Spirit (Willpower / Morale / Resonance)

    def to_dict(self) -> Dict[str, float]:
        return {
            "STR": self.STR,
            "VIT": self.VIT,
            "INT": self.INT,
            "AGI": self.AGI,
            "SPR": self.SPR,
        }


@dataclass
class SkillNode:
    skill_id: str
    name: str
    layer: SkillLayer
    description: str
    target_slot: SlotType
    hp_cost: float = 0.0
    mp_cost: float = 0.0
    stamina_cost: float = 0.0
    cooldown: float = 1.0  # Seconds or Ticks
    stat_requirements: Optional[Dict[str, float]] = None

    # Impact vector parameters (Friction, Karma/Reputation impact, Damage, Void impact)
    physical_friction: float = 0.0   # Physical force / damage output
    aggressiveness: float = 0.0      # Aggressive friction (+ for destructive, - for altruistic)
    reputation_shift: float = 0.0    # Direct lawful/outlaw reputation impact (-: outlaw, +: hero)
    resource_delta: Dict[str, float] = field(default_factory=dict) # Material transfer impact


@dataclass
class ActionExecutionResult:
    success: bool
    skill_id: str
    slot: SlotType
    caster_id: str
    target_id: Optional[str]
    message: str
    friction_generated: float
    reputation_delta: float
    aggressiveness: float
    hp_consumed: float
    mp_consumed: float
    stamina_consumed: float
    destructed_items: List[str] = field(default_factory=list)
    state_transitions: Dict[str, Any] = field(default_factory=dict)


class Entity:
    """
    Universal Entity class representing any actor in the world (Player, NPC, Guard, Boss).
    All entities interact through the exact same QWER protocol.
    """
    def __init__(
        self,
        entity_id: str,
        name: str,
        origin: str = "Human",
        bloodline: str = "Commoner",
        guild: str = "Unaffiliated",
        job: str = "Wanderer",
        base_stats: Optional[Stats] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.origin = origin
        self.bloodline = bloodline
        self.guild = guild
        self.job = job

        self.stats = base_stats if base_stats else Stats()

        # Max Capacities
        self.max_hp = self.stats.VIT * 20.0
        self.max_mp = self.stats.INT * 15.0
        self.max_stamina = (self.stats.STR + self.stats.AGI) * 10.0

        # Current Capacities
        self.current_hp = self.max_hp
        self.current_mp = self.max_mp
        self.current_stamina = self.max_stamina

        # Position (2D or 3D vector coordinates)
        self.position: Tuple[float, float] = (0.0, 0.0)

        # Inventory / Material Assets
        self.wealth: float = 100.0  # Gold/Resource units
        self.inventory: List[Dict[str, Any]] = [
            {"id": f"{entity_id}_starter_weapon", "name": "Basic Gear", "value": 50.0, "durability": 100.0}
        ]

        # Reputation & Outlaw State
        self.reputation_score: float = 0.0  # 0: Neutral, >50: Hero, <-50: Outlaw
        self.is_outlaw: bool = False
        self.is_dead: bool = False

        # QWER Skill Slots
        self.equipped_skills: Dict[SlotType, Optional[SkillNode]] = {
            SlotType.Q: None,
            SlotType.W: None,
            SlotType.E: None,
            SlotType.R: None
        }

        # Cooldown Timers (slot -> last_used_time). Initialized to -999.0 so skills are ready at t>=0.
        self.cooldown_timers: Dict[SlotType, float] = {
            SlotType.Q: -999.0,
            SlotType.W: -999.0,
            SlotType.E: -999.0,
            SlotType.R: -999.0
        }

        # Unlocked Skill Tree Nodes
        self.unlocked_skills: List[SkillNode] = []

    def equip_skill(self, skill: SkillNode) -> bool:
        """Equip a skill into its target QWER slot."""
        self.equipped_skills[skill.target_slot] = skill
        if skill not in self.unlocked_skills:
            self.unlocked_skills.append(skill)
        return True

    def update_reputation(self, delta: float):
        """Update reputation score and handle state transition between Hero and Outlaw."""
        self.reputation_score += delta
        if self.reputation_score <= -50.0 and not self.is_outlaw:
            self.is_outlaw = True
        elif self.reputation_score >= -10.0 and self.is_outlaw:
            self.is_outlaw = False

    def receive_damage(self, damage: float, attacker_id: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Receive physical friction / damage.
        Returns (is_dead, dropped_loot/destruction_products).
        """
        if self.is_dead:
            return True, []

        effective_damage = max(1.0, damage - (self.stats.VIT * 0.5))
        self.current_hp = max(0.0, self.current_hp - effective_damage)

        dropped_items = []
        if self.current_hp <= 0.0:
            self.is_dead = True
            # Full Loot & Destruction Law:
            # On death, inventory and wealth drop into the world or shatter, leaving a Void in entity capability.
            dropped_items = self.inventory.copy()
            self.inventory.clear()
            dropped_items.append({"id": "wealth_drop", "name": "Gold Coins", "value": self.wealth})
            self.wealth = 0.0

        return self.is_dead, dropped_items

    def execute_qwer_action(
        self,
        slot: SlotType,
        target_entity: Optional['Entity'] = None,
        current_time: float = 0.0
    ) -> ActionExecutionResult:
        """
        Universal Action Execution Cycle:
        [Input (Slot) -> Check Cooldown & Resources -> Consume -> Friction -> State Transition]
        """
        if self.is_dead:
            return ActionExecutionResult(
                success=False, skill_id="NONE", slot=slot, caster_id=self.entity_id,
                target_id=target_entity.entity_id if target_entity else None,
                message="Entity is dead.", friction_generated=0.0, reputation_delta=0.0,
                aggressiveness=0.0, hp_consumed=0.0, mp_consumed=0.0, stamina_consumed=0.0
            )

        skill = self.equipped_skills.get(slot)
        if not skill:
            return ActionExecutionResult(
                success=False, skill_id="EMPTY", slot=slot, caster_id=self.entity_id,
                target_id=target_entity.entity_id if target_entity else None,
                message=f"No skill equipped in slot {slot.value}.", friction_generated=0.0,
                reputation_delta=0.0, aggressiveness=0.0, hp_consumed=0.0, mp_consumed=0.0, stamina_consumed=0.0
            )

        # Check Cooldown
        last_used = self.cooldown_timers.get(slot, 0.0)
        if current_time - last_used < skill.cooldown:
            remaining = skill.cooldown - (current_time - last_used)
            return ActionExecutionResult(
                success=False, skill_id=skill.skill_id, slot=slot, caster_id=self.entity_id,
                target_id=target_entity.entity_id if target_entity else None,
                message=f"Skill {skill.name} is on cooldown ({remaining:.1f}s remaining).",
                friction_generated=0.0, reputation_delta=0.0, aggressiveness=0.0,
                hp_consumed=0.0, mp_consumed=0.0, stamina_consumed=0.0
            )

        # Check Resources
        if self.current_hp < skill.hp_cost or self.current_mp < skill.mp_cost or self.current_stamina < skill.stamina_cost:
            return ActionExecutionResult(
                success=False, skill_id=skill.skill_id, slot=slot, caster_id=self.entity_id,
                target_id=target_entity.entity_id if target_entity else None,
                message=f"Insufficient resources to cast {skill.name}.",
                friction_generated=0.0, reputation_delta=0.0, aggressiveness=0.0,
                hp_consumed=0.0, mp_consumed=0.0, stamina_consumed=0.0
            )

        # Consume Resources
        self.current_hp -= skill.hp_cost
        self.current_mp -= skill.mp_cost
        self.current_stamina -= skill.stamina_cost
        self.cooldown_timers[slot] = current_time

        # Calculate Physical Friction Output (Scaled by relevant stats)
        friction = skill.physical_friction * (1.0 + (self.stats.STR / 20.0))

        # Apply Damage / Friction to Target if present
        dropped_loot = []
        state_transitions = {}
        if target_entity and friction > 0.0:
            is_dead, loot = target_entity.receive_damage(friction, attacker_id=self.entity_id)
            dropped_loot = [item["name"] for item in loot]
            state_transitions["target_damaged"] = True
            state_transitions["target_died"] = is_dead

        # Update Caster's Reputation and State
        self.update_reputation(skill.reputation_shift)
        state_transitions["caster_is_outlaw"] = self.is_outlaw
        state_transitions["caster_reputation"] = self.reputation_score

        return ActionExecutionResult(
            success=True,
            skill_id=skill.skill_id,
            slot=slot,
            caster_id=self.entity_id,
            target_id=target_entity.entity_id if target_entity else None,
            message=f"{self.name} executed [{slot.value}] {skill.name}!",
            friction_generated=friction,
            reputation_delta=skill.reputation_shift,
            aggressiveness=skill.aggressiveness,
            hp_consumed=skill.hp_cost,
            mp_consumed=skill.mp_cost,
            stamina_consumed=skill.stamina_cost,
            destructed_items=dropped_loot,
            state_transitions=state_transitions
        )


def build_default_layered_skillset() -> Dict[str, SkillNode]:
    """
    Creates standard default skills across the 4 layers (Origin, Bloodline, Guild, Job).
    """
    skills = {}

    # --- Q Slot Skills (Job Layer Focus) ---
    skills["warrior_heavy_strike"] = SkillNode(
        skill_id="warrior_heavy_strike",
        name="Heavy Strike (강타)",
        layer=SkillLayer.JOB,
        description="A powerful physical blow that exerts high physical friction.",
        target_slot=SlotType.Q,
        stamina_cost=15.0,
        cooldown=1.0,
        physical_friction=30.0,
        aggressiveness=0.5,
        reputation_shift=-2.0
    )

    skills["merchant_escort_shield"] = SkillNode(
        skill_id="merchant_escort_shield",
        name="Trade Guard Barrier (상단 보호막)",
        layer=SkillLayer.JOB,
        description="Deploys a defensive barrier using wealth reserves.",
        target_slot=SlotType.Q,
        mp_cost=10.0,
        cooldown=2.0,
        physical_friction=5.0,
        aggressiveness=-0.2,
        reputation_shift=5.0
    )

    # --- W Slot Skills (Bloodline Layer Focus) ---
    skills["swordmaster_pacheonbo"] = SkillNode(
        skill_id="swordmaster_pacheonbo",
        name="Heritage Secret: Pacheonbo (비기: 파천보)",
        layer=SkillLayer.BLOODLINE,
        description="Family secret technique that lunges forward, bypassing defense.",
        target_slot=SlotType.W,
        stamina_cost=25.0,
        cooldown=3.0,
        physical_friction=50.0,
        aggressiveness=0.8,
        reputation_shift=-5.0
    )

    skills["shadow_stealth_step"] = SkillNode(
        skill_id="shadow_stealth_step",
        name="Shadow Stride (은영보)",
        layer=SkillLayer.BLOODLINE,
        description="Covert mobility move that reduces aggro and evades guards.",
        target_slot=SlotType.W,
        stamina_cost=20.0,
        cooldown=4.0,
        physical_friction=0.0,
        aggressiveness=-0.5,
        reputation_shift=-10.0  # Suspicions raised
    )

    # --- E Slot Skills (Guild Layer Focus) ---
    skills["knight_salute_mandate"] = SkillNode(
        skill_id="knight_salute_mandate",
        name="Knight Mandate (경비 경례)",
        layer=SkillLayer.GUILD,
        description="Uses official guild authority to pacify local aggro.",
        target_slot=SlotType.E,
        mp_cost=15.0,
        cooldown=5.0,
        physical_friction=0.0,
        aggressiveness=-0.8,
        reputation_shift=15.0
    )

    skills["blackmarket_covert_smuggle"] = SkillNode(
        skill_id="blackmarket_covert_smuggle",
        name="Black Market Stash (지하 밀거래)",
        layer=SkillLayer.GUILD,
        description="Access illicit resources while staying off the law grid.",
        target_slot=SlotType.E,
        mp_cost=20.0,
        cooldown=6.0,
        physical_friction=0.0,
        aggressiveness=0.3,
        reputation_shift=-15.0
    )

    # --- R Slot Skills (Origin Layer Ultimate Focus) ---
    skills["demon_frenzy_protocol"] = SkillNode(
        skill_id="demon_frenzy_protocol",
        name="Demonic Overclock (혈마 폭주)",
        layer=SkillLayer.ORIGIN,
        description="Burns HP instead of MP to unleash catastrophic dark friction.",
        target_slot=SlotType.R,
        hp_cost=40.0,
        cooldown=10.0,
        physical_friction=120.0,
        aggressiveness=1.0,
        reputation_shift=-30.0
    )

    skills["human_unity_resonance"] = SkillNode(
        skill_id="human_unity_resonance",
        name="Call of Humanity (의지의 결집)",
        layer=SkillLayer.ORIGIN,
        description="Rallies surrounding allies and boosts defense based on reputation.",
        target_slot=SlotType.R,
        mp_cost=50.0,
        cooldown=12.0,
        physical_friction=10.0,
        aggressiveness=-0.9,
        reputation_shift=25.0
    )

    return skills
