"""
[VERIFICATION: CLIFFORD GEOMETRIC ALGEBRA DYNAMICS]
Verifies Cl(p,q) multivector algebraic operations and quaternion isomorphism.
"""

import math
import sys
import os

# Ensure import paths are resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.math_utils import Quaternion, Multivector

def test_basis_multiplication():
    """Verify basic blade multiplication rules in Cl(3, 0)."""
    signature = (3, 0)
    # Basis vectors: e1 (1), e2 (2), e3 (4)
    e1 = Multivector({1: 1.0}, signature)
    e2 = Multivector({2: 1.0}, signature)
    e3 = Multivector({4: 1.0}, signature)

    # 1. Squares: e_i^2 = 1 in Euclidean space Cl(3,0)
    e1_sq = e1 * e1
    e2_sq = e2 * e2
    e3_sq = e3 * e3
    assert e1_sq.data == {0: 1.0}
    assert e2_sq.data == {0: 1.0}
    assert e3_sq.data == {0: 1.0}

    # 2. Anti-commutativity: e1 * e2 = - e2 * e1 = e12 (mask 3)
    e12 = e1 * e2
    e21 = e2 * e1
    assert e12.data == {3: 1.0}
    assert e21.data == {3: -1.0}

    # 3. Triple product: e1 * e2 * e3 = e123 (mask 7)
    e123 = e1 * e2 * e3
    assert e123.data == {7: 1.0}

    # 4. Reduction: e12 * e2 = e1 * e2 * e2 = e1
    e12_2 = e12 * e2
    assert e12_2.data == {1: 1.0}


def test_signature_negative():
    """Verify negative signature Cl(0, 3) where e_i^2 = -1."""
    signature = (0, 3)
    e1 = Multivector({1: 1.0}, signature)
    e2 = Multivector({2: 1.0}, signature)

    e1_sq = e1 * e1
    e2_sq = e2 * e2
    assert e1_sq.data == {0: -1.0}
    assert e2_sq.data == {0: -1.0}

    # e1 * e2 = e12, e2 * e1 = -e12 (anti-commutativity still holds)
    e12 = e1 * e2
    assert e12.data == {3: 1.0}
    
    # e12 * e2 = e1 * e2 * e2 = e1 * (-1) = -e1
    e12_2 = e12 * e2
    assert e12_2.data == {1: -1.0}


def test_wedge_and_dot_products():
    """Verify wedge (outer) and contraction (inner) products in Cl(3, 0)."""
    signature = (3, 0)
    e1 = Multivector({1: 1.0}, signature)
    e2 = Multivector({2: 1.0}, signature)
    
    # 1. Wedge Product (^)
    # e1 ^ e2 = e12
    e1_wedge_e2 = e1 ^ e2
    assert e1_wedge_e2.data == {3: 1.0}
    
    # e1 ^ e1 = 0
    e1_wedge_e1 = e1 ^ e1
    assert e1_wedge_e1.data == {}

    # 2. Contraction (Inner Product)
    # e1 . e1 = 1
    e1_dot_e1 = e1.dot(e1)
    assert e1_dot_e1.data == {0: 1.0}

    # e1 . e2 = 0
    e1_dot_e2 = e1.dot(e2)
    assert e1_dot_e2.data == {}

    # e1 . e12 = e2
    e12 = e1 * e2
    e1_dot_e12 = e1.dot(e12)
    assert e1_dot_e12.data == {2: 1.0}


def test_inverse_algebra():
    """Verify inverse operation in Cl(3, 0) for Even Subalgebra."""
    signature = (3, 0)
    # A = 2.0 + 1.5 * e12 + 0.8 * e23 (scalar and bivector combinations, invertible)
    # e12 is mask 3, e23 is mask 6
    a = Multivector({0: 2.0, 3: 1.5, 6: 0.8}, signature)
    a_inv = a.inverse()

    # A * A_inv should be close to 1.0 scalar
    prod = a * a_inv
    assert len(prod.data) == 1
    assert abs(prod.data[0] - 1.0) < 1e-7


def test_quaternion_isomorphism():
    """
    Verify isomorphism between Quaternion and Cl(3,0) Even Subalgebra.
    Isomorphism map:
      1 <-> 1 (0)
      i <-> -e23 (-6)
      j <-> -e31 (-5)
      k <-> -e12 (-3)
    """
    signature = (3, 0)

    # Let's test random coefficients
    w1, x1, y1, z1 = 1.2, -0.5, 0.8, 2.1
    w2, x2, y2, z2 = 0.7, 1.4, -0.9, 0.3

    q1 = Quaternion(w1, x1, y1, z1)
    q2 = Quaternion(w2, x2, y2, z2)

    # Hamilton Product Result
    q_prod = q1 * q2

    # Map to Cl(3,0) Multivectors
    # e1=1, e2=2, e3=4
    # e12 = 3, e23 = 6, e31 = 5 (e13 is 5 but order e3*e1 is -e13 = e31)
    # Note on e31: e3 (4) * e1 (1) -> swaps: 4*1 -> 4 has lower bit? No, 1 is lower than 4, so swaps = 1.
    # Therefore e3 * e1 = - e13. Under bitmask, 1 ^ 4 = 5 (e13). So mask 5 represents e13, and its value is -e31.
    # Therefore, -y * e31 = y * e13 (mask 5).
    # Let's write the exact mapping carefully:
    # i <-> -e23 (mask 6, coeff -x)
    # j <-> -e31 = e13 (mask 5, coeff +y)
    # k <-> -e12 (mask 3, coeff -z)
    
    mv1 = Multivector({0: w1, 6: -x1, 5: y1, 3: -z1}, signature)
    mv2 = Multivector({0: w2, 6: -x2, 5: y2, 3: -z2}, signature)

    mv_prod = mv1 * mv2

    # Map product back to quaternion elements
    res_w = mv_prod.data.get(0, 0.0)
    res_x = -mv_prod.data.get(6, 0.0)
    res_y = mv_prod.data.get(5, 0.0)
    res_z = -mv_prod.data.get(3, 0.0)

    assert abs(res_w - q_prod.w) < 1e-7
    assert abs(res_x - q_prod.x) < 1e-7
    assert abs(res_y - q_prod.y) < 1e-7
    assert abs(res_z - q_prod.z) < 1e-7
