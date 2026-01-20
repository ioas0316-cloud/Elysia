"""
Molecular Database
==================
Stores the fundamental physical properties of matter.
Used by the Providence Engine to determine State based on Energy.

Properties:
- atomic_mass: Relative mass (H=1)
- bond_energy_solid: Energy required to break lattice (eV) -> Melt
- bond_energy_liquid: Energy required to break surface tension (eV) -> Boil
- boiling_point_k: Kelvin temperature where Liquid -> Gas
"""

MOLECULAR_DATA = {
    "water": {
        "atomic_mass": 18.0,
        "bond_energy_solid": 0.06,  # Hydrogen bonds (Ice)
        "bond_energy_liquid": 0.42, # Vaporization energy
        "boiling_point_k": 373.15,  # 100°C
        "freezing_point_k": 273.15  # 0°C
    },
    "gold": {
        "atomic_mass": 197.0,
        "bond_energy_solid": 3.7,   # Metallic bond (Melting)
        "bond_energy_liquid": 3.2,  # Vaporization
        "boiling_point_k": 3129.0,
        "freezing_point_k": 1337.33 # Melting point
    },
    "iron": {
        "atomic_mass": 55.8,
        "bond_energy_solid": 4.0,
        "bond_energy_liquid": 3.8,
        "boiling_point_k": 3134.0,
        "freezing_point_k": 1811.0
    },
    "air": {
        "atomic_mass": 29.0,
        "bond_energy_solid": 0.01,
        "bond_energy_liquid": 0.01,
        "boiling_point_k": 77.0,   # Nitrogen approximation
        "freezing_point_k": 63.0
    }
}

def get_molecule(name: str):
    return MOLECULAR_DATA.get(name.lower(), {
        "atomic_mass": 10.0,
        "bond_energy_solid": 1.0, 
        "bond_energy_liquid": 1.0,
        "boiling_point_k": 1000.0,
        "freezing_point_k": 500.0
    })
