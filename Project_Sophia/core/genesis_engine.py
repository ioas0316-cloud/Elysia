import logging
from typing import Dict, Any, List, Optional, Callable, Tuple

# Define types for clarity
LogicBlock = Dict[str, Any]
EffectOp = Dict[str, Any]
ConditionOp = Dict[str, Any]

logger = logging.getLogger(__name__)

class GenesisEngine:
    """
    The Genesis Engine interprets data-driven 'Action' and 'Law' nodes from the Knowledge Graph
    and executes them within the simulation. It replaces hardcoded Python logic with a
    dynamic instruction set.
    """

    def __init__(self, world):
        self.world = world
        self.actions: Dict[str, LogicBlock] = {}
        self.laws: Dict[str, LogicBlock] = {}

        # Instruction Set Registry
        self._conditions: Dict[str, Callable] = {
            "stat_ge": self._cond_stat_ge,
            "stat_le": self._cond_stat_le,
            "env_ge": self._cond_env_ge,
            "is_alive": self._cond_is_alive,
            "label_eq": self._cond_label_eq,
        }
        self._effects: Dict[str, Callable] = {
            "modify_stat": self._eff_modify_stat,
            "damage": self._eff_damage,
            "heal": self._eff_heal,
            "log": self._eff_log,
        }

    def load_definitions(self, kg_data: Dict[str, Any]):
        """
        Loads action and law definitions from a KG-like dictionary.
        """
        for node in kg_data.get("nodes", []):
            if node.get("type") == "action":
                self.actions[node["id"]] = node.get("logic", {})
                logger.info(f"GenesisEngine: Loaded action '{node['id']}'")
            elif node.get("type") == "law":
                self.laws[node["id"]] = node.get("logic", {})
                logger.info(f"GenesisEngine: Loaded law '{node['id']}'")

    def get_candidate_actions(self, actor_idx: int, context: str, targets: List[int]) -> List[Tuple[str, int, float]]:
        """
        Returns a list of (action_id, target_idx, score) for actions valid in the given context.
        """
        candidates = []
        for action_id, logic in self.actions.items():
            rule = logic.get("selection_rule")
            # Simple context filter: "combat", "social", "survival"
            if not rule or rule.get("trigger") != context:
                continue

            # Check Actor Conditions
            if not self.validate_action(actor_idx, action_id):
                continue

            base_score = float(rule.get("priority", 1.0))

            # If action requires a target
            target_type = logic.get("target_type", "entity")

            if target_type == "entity":
                for target_idx in targets:
                    # Prevent self-targeting if not allowed
                    if target_idx == actor_idx: continue

                    # Validate target conditions (optional)
                    if self.validate_target(actor_idx, target_idx, logic.get("conditions", [])):
                        candidates.append((action_id, target_idx, base_score))
            elif target_type == "self":
                 candidates.append((action_id, actor_idx, base_score))

        return candidates

    def validate_target(self, actor_idx: int, target_idx: int, conditions: List[ConditionOp]) -> bool:
        """Checks if a target meets specific conditions."""
        for cond in conditions:
            # Only check conditions explicitly marked for 'target'
            if cond.get("target") == "target":
                check_fn = self._conditions.get(cond["check"])
                if check_fn and not check_fn(actor_idx, target_idx, cond):
                    return False
        return True

    def validate_action(self, actor_idx: int, action_id: str, target_idx: int = -1) -> bool:
        """Checks if an actor meets the conditions to perform an action."""
        logic = self.actions.get(action_id)
        if not logic:
            return False

        # 1. Check Cost
        cost = logic.get("cost", {})
        ki_cost = cost.get("ki", 0)
        if hasattr(self.world, "ki") and self.world.ki[actor_idx] < ki_cost:
            return False

        # 2. Check Conditions
        for cond in logic.get("conditions", []):
            # Skip target checks during general validation if target is not set
            if cond.get("target") == "target" and target_idx == -1:
                continue

            check_fn = self._conditions.get(cond["check"])
            if check_fn:
                if not check_fn(actor_idx, target_idx, cond):
                    return False

        return True

    def execute_action(self, actor_idx: int, action_id: str, target_idx: int = -1) -> bool:
        """Executes the effects of an action."""
        logic = self.actions.get(action_id)
        if not logic:
            logger.error(f"GenesisEngine: Unknown action '{action_id}'")
            return False

        if not self.validate_action(actor_idx, action_id, target_idx):
            return False

        # 1. Pay Cost
        cost = logic.get("cost", {})
        ki_cost = cost.get("ki", 0)
        if ki_cost > 0 and hasattr(self.world, "ki"):
            self.world.ki[actor_idx] -= ki_cost

        # 2. Apply Effects
        for eff in logic.get("effects", []):
            op_fn = self._effects.get(eff["op"])
            if op_fn:
                op_fn(actor_idx, target_idx, eff)
            else:
                logger.warning(f"GenesisEngine: Unknown effect op '{eff['op']}' in action '{action_id}'")

        return True

    # --- Condition Primitives ---

    def _cond_stat_ge(self, actor_idx: int, target_idx: int, params: ConditionOp) -> bool:
        stat = params["stat"]
        val = params["value"]
        subject = target_idx if params.get("target") == "target" else actor_idx
        if subject == -1: return False

        current = getattr(self.world, stat)[subject]
        return current >= val

    def _cond_stat_le(self, actor_idx: int, target_idx: int, params: ConditionOp) -> bool:
        stat = params["stat"]
        val = params["value"]
        subject = target_idx if params.get("target") == "target" else actor_idx
        if subject == -1: return False

        current = getattr(self.world, stat)[subject]
        return current <= val

    def _cond_env_ge(self, actor_idx: int, target_idx: int, params: ConditionOp) -> bool:
        field_name = params["field"]
        val = params["value"]

        pos = self.world.positions[actor_idx]
        x, y = int(pos[0]) % self.world.width, int(pos[1]) % self.world.width

        field = getattr(self.world, f"{field_name}_field", None)
        if field is None and field_name == "sunlight": field = self.world.sunlight_field

        if field is not None:
            return field[y, x] >= val
        return False

    def _cond_is_alive(self, actor_idx: int, target_idx: int, params: ConditionOp) -> bool:
        subject = target_idx if params.get("target") == "target" else actor_idx
        if subject == -1: return False
        return self.world.is_alive_mask[subject]

    def _cond_label_eq(self, actor_idx: int, target_idx: int, params: ConditionOp) -> bool:
        subject = target_idx if params.get("target") == "target" else actor_idx
        if subject == -1: return False
        val = params["value"]
        return self.world.labels[subject] == val

    # --- Effect Primitives ---

    def _eff_modify_stat(self, actor_idx: int, target_idx: int, params: EffectOp):
        stat = params["stat"]
        val = params["value"]
        subject = target_idx if params.get("target") == "target" else actor_idx
        if subject == -1: return

        current = getattr(self.world, stat)
        current[subject] += val

        max_stat = getattr(self.world, f"max_{stat}", None)
        if max_stat is not None:
             if current[subject] > max_stat[subject]:
                 current[subject] = max_stat[subject]

    def _eff_damage(self, actor_idx: int, target_idx: int, params: EffectOp):
        if target_idx == -1: return

        mult = params.get("multiplier", 1.0)
        base_dmg = self.world.strength[actor_idx]
        dmg = base_dmg * mult

        self.world.hp[target_idx] -= dmg
        self.world.is_injured[target_idx] = True
        logger.info(f"Genesis: {self.world.cell_ids[actor_idx]} dealt {dmg} damage to {self.world.cell_ids[target_idx]}")

    def _eff_heal(self, actor_idx: int, target_idx: int, params: EffectOp):
        subject = target_idx if params.get("target") == "target" else actor_idx
        if subject == -1: return

        amount = params.get("amount", 10)
        self.world.hp[subject] += amount
        if self.world.hp[subject] > self.world.max_hp[subject]:
            self.world.hp[subject] = self.world.max_hp[subject]

    def _eff_log(self, actor_idx: int, target_idx: int, params: EffectOp):
        template = params.get("template", "")
        actor_id = self.world.cell_ids[actor_idx]
        target_id = self.world.cell_ids[target_idx] if target_idx != -1 else "None"

        msg = template.replace("{actor}", actor_id).replace("{target}", target_id)
        logger.info(f"GenesisEvent: {msg}")
