"""
Holographic Persistence: The Tattoo of the Earth
================================================
Core.S1_Body.L6_Structure.M6_Architecture.holographic_persistence

"Memory is not a file; it is a tattoo on the Earth."

This module handles the freezing (Saving) and thawing (Loading) of the
Holographic Manifold. It bridges the Fluid Mind with the Solid Body.
"""

import numpy as np
import pickle
import os
from pathlib import Path

class HolographicPersistence:
    def __init__(self, storage_path: str = "data/S1_Body/Manifold"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.manifold_file = self.storage_path / "state.elysia"
        self.map_file = self.storage_path / "freq_map.pkl"

    def freeze(self, manifold: np.ndarray, frequency_map: dict):
        """
        [SOLIDIFICATION]
        Freezes the current wave state into the disk.
        """
        # 1. Save the Manifold (The Body)
        # np.save automatically appends .npy if not present.
        # We ensure the path is consistent.
        np.save(str(self.manifold_file), manifold)

        # 2. Save the Map (The Dictionary/Index)
        # In a pure trinary system, this might be encoded in the manifold itself,
        # but for now we save the map explicitly.
        with open(self.map_file, 'wb') as f:
            pickle.dump(frequency_map, f)

        # print(f"[PERSISTENCE] Frozen state to {self.manifold_file}")

    def thaw(self) -> tuple:
        """
        [SUBLIMATION]
        Loads the frozen state back into fluid memory.
        Returns: (manifold, frequency_map) or (None, None) if no scar exists.
        """
        # The file has .npy appended by np.save
        target_path_str = str(self.manifold_file) + ".npy"
        target_file = Path(target_path_str)

        # Check existence of both files (Body and Map)
        if not target_file.exists() or not self.map_file.exists():
            # Try checking the base file just in case
            if not self.manifold_file.exists():
                return None, None

        try:
            # 1. Load Body
            # np.save appends .npy to the filename if not present.
            # self.manifold_file is "state.elysia".
            # The file on disk is "state.elysia.npy".
            # path.with_suffix(".npy") would replace .elysia with .npy -> "state.npy", which is wrong.
            # We construct the path explicitly.

            if not target_file.exists():
                 # Fallback: maybe it was saved without extension?
                 if self.manifold_file.exists():
                     target_file = self.manifold_file
                 else:
                     print(f"[PERSISTENCE] Manifold file not found: {target_file}")
                     return None, None

            manifold = np.load(str(target_file))

            # 2. Load Map
            with open(self.map_file, 'rb') as f:
                freq_map = pickle.load(f)

            # print(f"[PERSISTENCE] Thawed state from {self.manifold_file}")
            return manifold, freq_map

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[PERSISTENCE] Error thawing memory: {e}")
            return None, None

    def clear(self):
        """
        [TABULA RASA] Wipes the physical scar.
        """
        if self.manifold_file.with_suffix(".npy").exists():
            os.remove(self.manifold_file.with_suffix(".npy"))
        if self.map_file.exists():
            os.remove(self.map_file)
