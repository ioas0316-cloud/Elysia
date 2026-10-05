"""
Project Elysia: Unified 5D Phase Space Core Engine & One-Page Dashboard
========================================================================
A unified RPG system mapping Skills, Items, NPCs, Monsters, Resources,
and Environmental Potentials onto a shared 5-Dimensional Phase Space (R^5).

Inspired by Stone Librande's "One Page Design Philosophy", this engine compresses
complex cascading dynamics into a single, cohesive mathematical framework and
generates unified One-Page Design dashboard reports.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# =====================================================================
# 1. 5D Phase Space Axis Definitions & Utilities
# =====================================================================

AXIS_NAMES = [
    "X1: Will / Heat / Ferocity (Destruction)",
    "X2: Flex / Density / Agility (Fluidity)",
    "X3: Order / Structure / Pattern (Logic)",
    "X4: Origin / Life / Adaptation (Ecology)",
    "X5: Field / Resonance / Transcendent (Social/Magic)"
]

AXIS_SHORT_NAMES = ["X1:Will", "X2:Flex", "X3:Order", "X4:Origin", "X5:Field"]

def clip_vector(vec: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """Clips vector components to bounded phase limits [-1.0, +1.0]."""
    return np.clip(np.array(vec, dtype=float), min_val, max_val)


# =====================================================================
# 2. Subsystem Entities
# =====================================================================

class EnvironmentPotential:
    """
    Master Environmental Potential Field V_Env(x, t) in R^5.
    Controls macro Zeitgeist and local regional phase fields.
    """
    def __init__(self, name: str, base_vec: np.ndarray):
        self.name = name
        self.vector = clip_vector(base_vec)

    def shift_zeitgeist(self, delta_vec: np.ndarray) -> np.ndarray:
        """Simulates macro Zeitgeist shift in the realm."""
        self.vector = clip_vector(self.vector + np.array(delta_vec, dtype=float))
        return self.vector

    def get_potentials(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vector": self.vector.copy(),
            "vector_list": np.round(self.vector, 2).tolist()
        }


class ResourceEntity:
    """
    Mutatable Natural Resource (Herbs, Ores, Crystals, Energy Wells).
    Adapts and mutates based on coupling with environmental potential fields.
    """
    def __init__(self, name: str, base_R: np.ndarray):
        self.name = name
        self.base_R = clip_vector(base_R)
        self.current_R = np.copy(self.base_R)

    def adapt_to_environment(self, env_vec: np.ndarray, coupling: float = 0.5) -> np.ndarray:
        """Warps resource phase coordinates based on environmental potential V_Env."""
        self.current_R = clip_vector(self.base_R + coupling * np.array(env_vec, dtype=float))
        return self.current_R

    def get_description(self) -> str:
        heat, density, struct, origin, res = self.current_R
        tags = []
        if heat > 0.4: tags.append("화염이 깃든")
        elif heat < -0.4: tags.append("서리 얼어붙은")

        if density > 0.4: tags.append("고밀도")
        elif density < -0.4: tags.append("희소 유동성")

        if struct > 0.4: tags.append("정제된 결정성")
        if origin > 0.4: tags.append("원시 생명력의")
        if res > 0.4: tags.append("위상 공명하는")

        prefix = " ".join(tags) if tags else "일반적인"
        return f"[{prefix} {self.name}]"

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.get_description(),
            "phase_vector": np.round(self.current_R, 2).tolist()
        }


class CraftedItemEntity:
    """
    Crafted Item derived from resource superposition & quantum energy shell discretization.
    """
    TIER_LABELS = {1: "Common", 2: "Rare/Epic", 3: "Legendary", 4: "Mythic Sovereign"}

    def __init__(self, name: str, resource_vectors: List[np.ndarray], energy_shell_n: int = 1, base_E0: float = 1000.0):
        self.name = name
        self.I_vector = clip_vector(np.mean(resource_vectors, axis=0))
        self.energy_shell_n = max(1, energy_shell_n)
        self.base_E0 = base_E0

    def calculate_quantum_leap_cost(self) -> float:
        """
        Calculates quantum energy required for critical tier leap:
        Delta E = E0 * (1 / n^2 - 1 / (n + 1)^2)
        """
        n = self.energy_shell_n
        return self.base_E0 * ((1.0 / (n ** 2)) - (1.0 / ((n + 1) ** 2)))

    def upgrade_tier(self) -> int:
        """Upgrades quantum energy shell tier."""
        self.energy_shell_n += 1
        return self.energy_shell_n

    def get_tier_label(self) -> str:
        return self.TIER_LABELS.get(self.energy_shell_n, f"Tier-{self.energy_shell_n}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.get_tier_label(),
            "quantum_shell_n": self.energy_shell_n,
            "phase_vector": np.round(self.I_vector, 2).tolist(),
            "quantum_leap_cost": round(self.calculate_quantum_leap_cost(), 1)
        }


class MonsterEntity:
    """
    Ecosystem Monster Entity subject to environmental mutation & phase shift bifurcation.
    """
    def __init__(self, name: str, base_M: np.ndarray, energy_shell: int = 1):
        self.name = name
        self.base_M = clip_vector(base_M)
        self.current_M = np.copy(self.base_M)
        self.energy_shell = energy_shell  # 1: Normal, 2: Elite, 3: World Boss

    def mutate(self, env_vec: np.ndarray, env_weight: float = 0.6) -> np.ndarray:
        """Mutates monster ecology vector toward environmental potential field."""
        self.current_M = clip_vector(self.base_M * (1.0 - env_weight) + np.array(env_vec, dtype=float) * env_weight)
        return self.current_M

    def get_status(self) -> Dict[str, Any]:
        ferocity, agility, struct, adapt, res = self.current_M

        if self.energy_shell >= 3:
            if struct > 0.4 and ferocity > 0.4:
                pattern = "3페이즈: 시공간 파괴 전장 왜곡 및 광폭화"
            elif ferocity > 0.3:
                pattern = "2페이즈: 단순 광폭 난무"
            else:
                pattern = "1페이즈: 수비적 방어 기믹"
            tier_title = "지역 월드 보스"
        elif self.energy_shell == 2:
            pattern = "엘리트 오라 및 유닛 소환"
            tier_title = "네임드 엘리트"
        else:
            pattern = "기본 추적 및 단일 타격"
            tier_title = "일반 개체"

        return {
            "title": f"{tier_title} [{self.name}]",
            "tier_level": self.energy_shell,
            "phase_vector": np.round(self.current_M, 2).tolist(),
            "pattern": pattern
        }


class NPCEntity:
    """
    Social Persona entity with cognitive dissonance stress tensor dynamics.
    """
    def __init__(self, name: str, base_S: np.ndarray, relaxation_rate: float = 0.15):
        self.name = name
        self.S = clip_vector(base_S)
        self.dissonance = 0.0
        self.relaxation_rate = relaxation_rate

    def evaluate_dissonance(self, env_vec: np.ndarray) -> float:
        """
        Calculates stress tensor norm (cognitive dissonance) against environment,
        then relaxes phase vector S toward environmental potential.
        """
        target = np.array(env_vec, dtype=float)
        self.dissonance = float(np.linalg.norm(self.S - target))
        self.S = clip_vector(self.S + self.relaxation_rate * (target - self.S))
        return self.dissonance

    def get_speech_pattern(self) -> str:
        will, flex, order, origin, field = self.S
        if order > 0.4 and origin > 0.3:
            tone = "격식체 고어 ('고대의 법도와 맹세를 준수하겠나이다.')"
        elif will > 0.4 and flex > 0.3:
            tone = "실리적 거친 언사 ('명분은 필요 없다. 당장 눈앞의 실속만 챙겨.')"
        elif field > 0.5:
            tone = "선동적 공명체 ('모두 들으라! 시대의 대변혁이 눈앞에 다가왔다!')"
        elif flex < -0.4 and order < -0.4:
            tone = "방랑자적 야성체 ('바람의 흐름을 따라 거닐 뿐.')"
        else:
            tone = "평범하고 절제된 중립어"

        if self.dissonance > 1.0:
            tone += " [경고: 심각한 사상적 이탈 및 내적 광기 표출]"
        return tone

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase_vector": np.round(self.S, 2).tolist(),
            "dissonance": round(self.dissonance, 3),
            "speech_pattern": self.get_speech_pattern()
        }


# =====================================================================
# 3. Unified Engine & One-Page Dashboard
# =====================================================================

class ElysiaUnifiedEngine:
    """
    Unified 5D Phase Space Engine coordinating Environment, Resources,
    Crafted Items, Monsters, and NPCs in a cascading relaxation dynamics loop.
    """
    def __init__(self, env_name: str = "신정정치 평화기", env_base_vec: Optional[np.ndarray] = None):
        if env_base_vec is None:
            env_base_vec = np.array([-0.4, -0.5, 0.8, 0.8, 0.2])
        self.env = EnvironmentPotential(env_name, env_base_vec)
        self.resources: List[ResourceEntity] = []
        self.crafted_items: List[CraftedItemEntity] = []
        self.monsters: List[MonsterEntity] = []
        self.npcs: List[NPCEntity] = []

    def add_resource(self, resource: ResourceEntity) -> None:
        self.resources.append(resource)

    def add_crafted_item(self, item: CraftedItemEntity) -> None:
        self.crafted_items.append(item)

    def add_monster(self, monster: MonsterEntity) -> None:
        self.monsters.append(monster)

    def add_npc(self, npc: NPCEntity) -> None:
        self.npcs.append(npc)

    def step_cascade(self, delta_env: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Triggers cascading relaxation across all subsystems when environmental potential shifts.
        """
        if delta_env is not None:
            self.env.shift_zeitgeist(delta_env)

        env_vec = self.env.vector

        # 1. Resource adaptation
        for res in self.resources:
            res.adapt_to_environment(env_vec)

        # 2. Re-evaluate crafted item superposition if resources are bound
        # (Items update phase vector from current resource states)

        # 3. Monster mutation
        for mon in self.monsters:
            mon.mutate(env_vec)

        # 4. NPC cognitive dissonance & relaxation
        for npc in self.npcs:
            npc.evaluate_dissonance(env_vec)

        return self.get_system_snapshot()

    def get_system_snapshot(self) -> Dict[str, Any]:
        return {
            "environment": self.env.get_potentials(),
            "resources": [r.get_status() for r in self.resources],
            "crafted_items": [i.get_status() for i in self.crafted_items],
            "monsters": [m.get_status() for m in self.monsters],
            "npcs": [n.get_status() for n in self.npcs]
        }

    def render_one_page_dashboard(self, title_suffix: str = "") -> str:
        """
        Renders a terminal One-Page Design Dashboard Report summarizing
        the entire 5D Phase Space system state in a clear single-page layout.
        Inspired by Stone Librande's "One Page Design Philosophy".
        """
        snapshot = self.get_system_snapshot()
        env_data = snapshot["environment"]
        env_vec = env_data["vector_list"]

        lines = []
        lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
        lines.append(f"│ PROJECT ELYSIA :: ONE-PAGE SYSTEM DESIGN DASHBOARD {title_suffix:<25} │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│ [MACRO ZEITGEIST POTENTIAL FIELD (V_Env)]                                   │")
        lines.append(f"│  Region/Era : {env_data['name']:<60} │")
        lines.append(f"│  5D Vector  : X1:{env_vec[0]:+0.2f} | X2:{env_vec[1]:+0.2f} | X3:{env_vec[2]:+0.2f} | X4:{env_vec[3]:+0.2f} | X5:{env_vec[4]:+0.2f}      │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [SUBSYSTEM 1: NATURAL RESOURCES (Ecology Adaptation)]                        │")
        if snapshot["resources"]:
            for r in snapshot["resources"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in r["phase_vector"]])
                lines.append(f"│  • {r['description']:<35} | Phase: [{vec_str}] │")
        else:
            lines.append("│  (No registered resources)                                                   │")

        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [SUBSYSTEM 2: CRAFTED ITEMS (Resource Superposition & Quantum Shells)]       │")
        if snapshot["crafted_items"]:
            for item in snapshot["crafted_items"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in item["phase_vector"]])
                lines.append(f"│  • {item['name']} ({item['tier']}) | Leap Cost: {item['quantum_leap_cost']:>6.1f}J | [{vec_str}] │")
        else:
            lines.append("│  (No registered crafted items)                                              │")

        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [SUBSYSTEM 3: MONSTERS & ECOSYSTEM (Ecology Mutation & Phase Shift)]         │")
        if snapshot["monsters"]:
            for m in snapshot["monsters"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in m["phase_vector"]])
                lines.append(f"│  • {m['title']:<25} | [{vec_str}]                        │")
                lines.append(f"│    └ Pattern: {m['pattern']:<57} │")
        else:
            lines.append("│  (No registered monsters)                                                    │")

        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│ [SUBSYSTEM 4: NPCS & SOCIAL PERSONA (Cognitive Dissonance Tensor)]           │")
        if snapshot["npcs"]:
            for n in snapshot["npcs"]:
                vec_str = ", ".join([f"{v:+0.2f}" for v in n["phase_vector"]])
                lines.append(f"│  • {n['name']} | Dissonance: {n['dissonance']:0.3f} | [{vec_str}]                  │")
                lines.append(f"│    └ Speech: {n['speech_pattern']:<58} │")
        else:
            lines.append("│  (No registered NPCs)                                                       │")

        lines.append("└──────────────────────────────────────────────────────────────────────────────┘")

        return "\n".join(lines)
