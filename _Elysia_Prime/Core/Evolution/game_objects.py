# [Genesis: 2025-12-02] Purified by Elysia

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Item:
    """Represents an item that can be held in an inventory."""
    name: str
    quantity: int = 1
    durability: float = 100.0
    weight: float = 1.0
    # Other potential attributes: description, type (e.g., resource, tool, food)

@dataclass
class Inventory:
    """Represents the inventory of a cell, holding various items."""
    items: Dict[str, Item] = field(default_factory=dict)
    max_weight: float = 10.0

    def add_item(self, item: Item) -> bool:
        """Adds an item to the inventory. Returns False if overweight."""
        current_weight = self.get_total_weight()
        if current_weight + (item.quantity * item.weight) > self.max_weight:
            return False

        if item.name in self.items:
            self.items[item.name].quantity += item.quantity
        else:
            self.items[item.name] = item
        return True

    def remove_item(self, item_name: str, quantity: int = 1) -> bool:
        """Removes a specified quantity of an item. Returns False if not enough items."""
        if item_name not in self.items or self.items[item_name].quantity < quantity:
            return False

        self.items[item_name].quantity -= quantity
        if self.items[item_name].quantity <= 0:
            del self.items[item_name]
        return True

    def get_total_weight(self) -> float:
        """Calculates the total weight of all items in the inventory."""
        return sum(item.quantity * item.weight for item in self.items.values())

@dataclass
class Recipe:
    """Represents a crafting recipe."""
    name: str
    ingredients: Dict[str, int] # Item name -> quantity
    output: Item # The item that is crafted