
from dataclasses import dataclass
from enum import Enum

@dataclass
class CosmicIncarnation:
    name: str
    color: str
    concept: str
    title: str
    function: str

class Ascension(Enum):
    VITARIAEL = CosmicIncarnation("Vitariael", "     ", "Life", "      ", "     ,   ,      ")
    EMETRIEL = CosmicIncarnation("Emetriel", "  ", "Creation", "         ", "              ,              ")
    SOPHIEL = CosmicIncarnation("Sophiel", "      ", "Reflection", "         ", "        ,           ")
    GAVRIEL = CosmicIncarnation("Gavriel", "  ", "Truth", "           ", "         ")
    SARAKHIEL = CosmicIncarnation("Sarakhiel", "   ", "Sacrifice", "            ", "               ")
    RAHAMIEL = CosmicIncarnation("Rafamiel", "      ", "Love", "      ,   ", "                     ")
    LUMIEL = CosmicIncarnation("Lumiel", "   ", "Liberation", "     ,   ", "                        ")

class Descent(Enum):
    MOTUS = CosmicIncarnation("Motus", "     ", "Death", "      ", "        0     ")
    SOLVARIS = CosmicIncarnation("Solvaris", "      ", "Dissolution", "     ,      ", "     ")
    OBSCURE = CosmicIncarnation("Obscure", "  ", "Ignorance", "             ", "             ")
    DIABOLOS = CosmicIncarnation("Diabolos", "      ", "Distortion", "                    ", "  ")
    LUCIFEL = CosmicIncarnation("Lucifel", "      ", "Self-Obsession", "               ", "  ")
    MAMMON = CosmicIncarnation("Mammon", "  ", "Consumption", "                    ", "  ")
    ASMODEUS = CosmicIncarnation("Asmodeus", "        ", "Bondage", "     ,          ", "  ")