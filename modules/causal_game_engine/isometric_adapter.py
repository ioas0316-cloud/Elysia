"""
Isometric Tilemap and Spatial Projection Adapter module for Elysia Causal Game Engine.

Handles 2.5D Isometric coordinate projections, multi-layer tilemap data representation,
and CC0 asset metadata mapping (e.g. Kenney Isometric Medieval / City asset packs).
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class IsoTileConfig:
    """Configuration for Isometric projection geometry."""
    tile_width: float = 64.0
    tile_height: float = 32.0
    elevation_scale: float = 16.0
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class AssetMetadata:
    """Metadata for CC0 isometric assets (Kenney format / OpenGameArt)."""
    asset_id: str
    name: str
    category: str  # 'terrain', 'building', 'resource', 'entity'
    sprite_path: str = ""
    width_tiles: int = 1
    height_tiles: int = 1
    anchor_offset_x: float = 0.0
    anchor_offset_y: float = 0.0
    passable: bool = True
    buildable: bool = True
    friction_coefficient: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IsoTileCell:
    """Cell in the isometric tilemap multi-layer grid."""
    grid_x: int
    grid_y: int
    elevation: float = 0.0
    terrain_asset_id: str = "grass"
    building_asset_id: Optional[str] = None
    resource_asset_id: Optional[str] = None
    entity_asset_ids: List[str] = field(default_factory=list)
    custom_friction: float = 1.0
    causal_potential: float = 0.0


class IsometricProjection:
    """
    Mathematical projection between 3D Discrete Grid Space (x, y, elevation)
    and 2D Isometric Viewport Space (iso_x, iso_y).
    """

    def __init__(self, config: Optional[IsoTileConfig] = None):
        self.config = config or IsoTileConfig()

    def grid_to_iso(self, grid_x: float, grid_y: float, elevation: float = 0.0) -> Tuple[float, float]:
        """
        Projects 3D Grid space (grid_x, grid_y, elevation) to 2D Isometric screen coordinates.

        Formula:
          iso_x = (grid_x - grid_y) * (tile_width / 2) + origin_x
          iso_y = (grid_x + grid_y) * (tile_height / 2) - (elevation * elevation_scale) + origin_y
        """
        w2 = self.config.tile_width / 2.0
        h2 = self.config.tile_height / 2.0

        iso_x = (grid_x - grid_y) * w2 + self.config.origin_x
        iso_y = (grid_x + grid_y) * h2 - (elevation * self.config.elevation_scale) + self.config.origin_y
        return iso_x, iso_y

    def iso_to_grid(self, iso_x: float, iso_y: float, elevation: float = 0.0) -> Tuple[float, float]:
        """
        Inverse projection from 2D Isometric screen coordinates to 2D Grid space (grid_x, grid_y),
        given a target elevation layer.

        Formula derived from solving grid_to_iso linear system:
          adj_y = iso_y - origin_y + (elevation * elevation_scale)
          adj_x = iso_x - origin_x
          grid_x = (adj_x / w2 + adj_y / h2) / 2
          grid_y = (adj_y / h2 - adj_x / w2) / 2
        """
        w2 = self.config.tile_width / 2.0
        h2 = self.config.tile_height / 2.0

        adj_x = iso_x - self.config.origin_x
        adj_y = iso_y - self.config.origin_y + (elevation * self.config.elevation_scale)

        grid_x = (adj_x / w2 + adj_y / h2) / 2.0
        grid_y = (adj_y / h2 - adj_x / w2) / 2.0
        return grid_x, grid_y


class CC0AssetRegistry:
    """
    Registry for CC0 Isometric Medieval / City asset packs metadata (e.g. Kenney.nl).
    """

    def __init__(self):
        self._assets: Dict[str, AssetMetadata] = {}
        self._load_default_kenney_pack()

    def register_asset(self, asset: AssetMetadata):
        self._assets[asset.asset_id] = asset

    def get_asset(self, asset_id: str) -> Optional[AssetMetadata]:
        return self._assets.get(asset_id)

    def _load_default_kenney_pack(self):
        """Pre-loads standard Kenney Isometric Medieval & City CC0 asset metadata."""
        defaults = [
            # Terrain
            AssetMetadata("grass", "Grass Tile", "terrain", passable=True, buildable=True, friction_coefficient=1.0),
            AssetMetadata("dirt_path", "Dirt Path Tile", "terrain", passable=True, buildable=True, friction_coefficient=0.8),
            AssetMetadata("water", "Water Tile", "terrain", passable=False, buildable=False, friction_coefficient=5.0),
            AssetMetadata("stone_pavement", "Stone Pavement", "terrain", passable=True, buildable=True, friction_coefficient=0.7),

            # Buildings
            AssetMetadata("town_hall", "Town Hall", "building", width_tiles=2, height_tiles=2, passable=False, buildable=False, properties={"capacity": 20, "role": "governance"}),
            AssetMetadata("house", "Medieval House", "building", width_tiles=1, height_tiles=1, passable=False, buildable=False, properties={"capacity": 5, "role": "shelter"}),
            AssetMetadata("granary", "Granary / Storehouse", "building", width_tiles=1, height_tiles=1, passable=False, buildable=False, properties={"capacity": 500, "role": "storage"}),
            AssetMetadata("farm_field", "Wheat Farm", "building", width_tiles=2, height_tiles=2, passable=True, buildable=False, friction_coefficient=1.2, properties={"yield_rate": 10.0, "role": "production"}),
            AssetMetadata("lumber_camp", "Lumber Camp", "building", width_tiles=1, height_tiles=1, passable=False, buildable=False, properties={"role": "production"}),

            # Resources
            AssetMetadata("tree_oak", "Oak Tree", "resource", passable=False, buildable=False, properties={"resource_type": "wood", "amount": 100}),
            AssetMetadata("gold_mine", "Gold Deposit", "resource", passable=False, buildable=False, properties={"resource_type": "gold", "amount": 500}),
            AssetMetadata("stone_quarry", "Stone Deposit", "resource", passable=False, buildable=False, properties={"resource_type": "stone", "amount": 300}),

            # Entities / Units
            AssetMetadata("villager", "Villager NPC", "entity", passable=True, buildable=True, properties={"speed": 1.5, "role": "worker"}),
            AssetMetadata("builder", "Builder NPC", "entity", passable=True, buildable=True, properties={"speed": 1.2, "role": "constructor"}),
        ]
        for asset in defaults:
            self.register_asset(asset)


class IsometricTilemap:
    """
    Multi-layer Isometric Tilemap representation for Elysia Causal Engine.
    Coordinates layers for Terrain, Buildings, Resources, and Entities over an NxM grid.
    """

    def __init__(self, width: int, height: int, projection_config: Optional[IsoTileConfig] = None):
        self.width = width
        self.height = height
        self.projection = IsometricProjection(projection_config)
        self.asset_registry = CC0AssetRegistry()
        self.grid: List[List[IsoTileCell]] = [
            [IsoTileCell(grid_x=x, grid_y=y) for y in range(height)]
            for x in range(width)
        ]

    def is_valid_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> Optional[IsoTileCell]:
        if self.is_valid_coord(x, y):
            return self.grid[x][y]
        return None

    def set_elevation(self, x: int, y: int, elevation: float):
        cell = self.get_cell(x, y)
        if cell:
            cell.elevation = elevation

    def set_building(self, x: int, y: int, building_asset_id: Optional[str]):
        cell = self.get_cell(x, y)
        if cell:
            cell.building_asset_id = building_asset_id

    def set_resource(self, x: int, y: int, resource_asset_id: Optional[str]):
        cell = self.get_cell(x, y)
        if cell:
            cell.resource_asset_id = resource_asset_id

    def compute_cell_friction(self, x: int, y: int) -> float:
        """Computes the combined movement/passability friction for a cell."""
        cell = self.get_cell(x, y)
        if not cell:
            return float('inf')

        # Check passability of terrain
        terrain_meta = self.asset_registry.get_asset(cell.terrain_asset_id)
        if terrain_meta and not terrain_meta.passable:
            return float('inf')

        # Check building passability
        if cell.building_asset_id:
            b_meta = self.asset_registry.get_asset(cell.building_asset_id)
            if b_meta and not b_meta.passable:
                return float('inf')

        # Check resource passability
        if cell.resource_asset_id:
            r_meta = self.asset_registry.get_asset(cell.resource_asset_id)
            if r_meta and not r_meta.passable:
                return float('inf')

        base_friction = cell.custom_friction
        if terrain_meta:
            base_friction *= terrain_meta.friction_coefficient

        return max(0.1, base_friction)
