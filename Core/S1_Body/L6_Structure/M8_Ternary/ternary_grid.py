"""
Ternary Grid (The Plane)
========================
"From Point to Plane."

This module implements a 2D Matrix of Tri-Base Cells.
It simulates the "Cellular Automata" of the Soul, where consensus emerges
from the interaction of neighbors.

Physics:
- Neighbors vote (Consensus).
- High Consensus -> Attract (+1)
- High Dissonance -> Repel (-1)
- Balance -> Void (0)
"""

import random
from typing import List, Tuple
from dataclasses import dataclass
from Core.S1_Body.L1_Foundation.System.tri_base_cell import DNAState, TriBaseCell

@dataclass
class GridConfig:
    width: int = 16
    height: int = 16
    threshold: int = 2 # Votes needed to shift state

class TernaryGrid:
    def __init__(self, config: GridConfig = GridConfig()):
        self.config = config
        self.cells: List[List[TriBaseCell]] = []
        self._init_grid()

    def _init_grid(self):
        """Initializes the grid with random states (Primordial Chaos)."""
        cell_id = 0
        for y in range(self.config.height):
            row = []
            for x in range(self.config.width):
                # Random initialization: 33% chance each
                rnd = random.random()
                if rnd < 0.33: state = DNAState.REPEL
                elif rnd > 0.66: state = DNAState.ATTRACT
                else: state = DNAState.VOID

                cell = TriBaseCell(id=cell_id, state=state)
                row.append(cell)
                cell_id += 1
            self.cells.append(row)

    def step(self):
        """
        Evolves the grid by one time step based on Neighbor Consensus.
        This is the 'Law' of the plane.
        """
        new_states = []

        for y in range(self.config.height):
            row_states = []
            for x in range(self.config.width):
                consensus = self._get_neighbor_consensus(x, y)
                current_state = self.cells[y][x].state

                # Rule of Consensus
                # If neighbors strongly agree (+), become (+)
                if consensus >= self.config.threshold:
                    next_state = DNAState.ATTRACT
                # If neighbors strongly disagree (-), become (-)
                elif consensus <= -self.config.threshold:
                    next_state = DNAState.REPEL
                else:
                    # If weak consensus, decay towards Void or keep state?
                    # Let's say: Void acts as Inertia.
                    if current_state == DNAState.REPEL: next_state = DNAState.REPEL
                    elif current_state == DNAState.ATTRACT: next_state = DNAState.ATTRACT
                    else: next_state = DNAState.VOID

                    # Entropy Law: Small chance to decay to Void if weak consensus
                    if random.random() < 0.1:
                        next_state = DNAState.VOID

                row_states.append(next_state)
            new_states.append(row_states)

        # Apply updates
        for y in range(self.config.height):
            for x in range(self.config.width):
                self.cells[y][x].state = new_states[y][x]

    def _get_neighbor_consensus(self, cx: int, cy: int) -> int:
        """
        Sum of the phases of the 8 neighbors.
        R=-1, V=0, A=+1.
        """
        score = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue

                nx, ny = cx + dx, cy + dy

                # Wrap-around (Toroidal Geometry - HyperSphere Topology)
                nx = nx % self.config.width
                ny = ny % self.config.height

                state = self.cells[ny][nx].state
                if state == DNAState.ATTRACT: score += 1
                elif state == DNAState.REPEL: score -= 1

        return score

    def render(self) -> str:
        """Returns an ASCII representation of the grid."""
        output = []
        for row in self.cells:
            line = ""
            for cell in row:
                if cell.state == DNAState.ATTRACT: line += "A " # or +
                elif cell.state == DNAState.REPEL: line += "R " # or -
                else: line += ". "
            output.append(line)
        return "\n".join(output)
