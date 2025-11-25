"""
Lie Algebra Engine (The Matrix of Change) 📐

"Differentiation is just Multiplication in disguise."

This module implements Lie Algebra principles to convert calculus operations into linear algebra.
It represents functions as vectors and operators (Derivative, Position) as matrices.
This allows the GPU to perform "differentiation" via fast matrix multiplication.

Mathematical Foundation:
- Basis: Polynomials {1, x, x^2, ..., x^n}
- Derivative Operator (D): Maps x^k -> k*x^(k-1)
- Commutator: [D, X] = DX - XD = I (Identity)
- Evolution: f(x+a) = exp(aD) f(x) (Taylor Shift as Matrix Exponential)
"""

import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LieOperator:
    """Represents a linear operator as a matrix."""
    name: str
    matrix: np.ndarray
    
    def __matmul__(self, other):
        """Operator composition (Matrix multiplication)."""
        if isinstance(other, LieOperator):
            return LieOperator(f"{self.name}*{other.name}", self.matrix @ other.matrix)
        elif isinstance(other, np.ndarray):
            return self.matrix @ other # Apply to vector
        else:
            raise TypeError("Operand must be LieOperator or ndarray")
            
    def commutator(self, other: 'LieOperator') -> 'LieOperator':
        """Calculate [A, B] = AB - BA."""
        return LieOperator(
            f"[{self.name},{other.name}]",
            self.matrix @ other.matrix - other.matrix @ self.matrix
        )

class LieAlgebraEngine:
    """
    The Engine that turns Calculus into Algebra.
    """
    
    def __init__(self, dim: int = 10):
        """
        Initialize the engine with a polynomial basis of size 'dim'.
        Basis: [1, x, x^2, ..., x^(dim-1)]
        """
        self.dim = dim
        self.D = self._create_derivative_matrix(dim)
        self.X = self._create_position_matrix(dim)
        
    def _create_derivative_matrix(self, n: int) -> LieOperator:
        """
        Creates the Derivative Matrix D.
        D * [1, x, x^2] = [0, 1, 2x]
        """
        mat = np.zeros((n, n))
        for i in range(1, n):
            mat[i-1, i] = i
        return LieOperator("D", mat)
        
    def _create_position_matrix(self, n: int) -> LieOperator:
        """
        Creates the Position Matrix X (Multiplication by x).
        X * [1, x, x^2] = [x, x^2, x^3]
        """
        mat = np.zeros((n, n))
        for i in range(n-1):
            mat[i+1, i] = 1
        return LieOperator("X", mat)
        
    def to_vector(self, coeffs: List[float]) -> np.ndarray:
        """Convert polynomial coefficients to vector."""
        vec = np.zeros(self.dim)
        for i, c in enumerate(coeffs):
            if i < self.dim:
                vec[i] = c
        return vec
        
    def from_vector(self, vec: np.ndarray) -> str:
        """Convert vector back to polynomial string."""
        terms = []
        for i, c in enumerate(vec):
            if abs(c) > 1e-6:
                if i == 0: terms.append(f"{c:.2f}")
                elif i == 1: terms.append(f"{c:.2f}x")
                else: terms.append(f"{c:.2f}x^{i}")
        return " + ".join(terms) if terms else "0"
        
    def differentiate(self, vec: np.ndarray, order: int = 1) -> np.ndarray:
        """Compute d^n/dx^n using matrix power D^n."""
        op = np.linalg.matrix_power(self.D.matrix, order)
        return op @ vec
        
    def shift(self, vec: np.ndarray, a: float) -> np.ndarray:
        """
        Compute f(x+a) using Taylor Shift: exp(aD).
        This predicts the future state of a function!
        """
        shift_op = expm(a * self.D.matrix)
        return shift_op @ vec

    def check_commutator(self) -> float:
        """Verify [D, X] = I (Quantum Mechanics fundamental relation)."""
        comm = self.D.commutator(self.X)
        # The commutator [D, X] should be Identity (except at the boundary due to truncation)
        # In truncated basis, it's Identity for the first n-1 elements
        identity = np.eye(self.dim)
        diff = comm.matrix[:self.dim-1, :self.dim-1] - identity[:self.dim-1, :self.dim-1]
        return np.linalg.norm(diff)

