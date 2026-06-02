"""
Elysia Core Mathematics Utility Library
=======================================
This library provides clean, optimized, and unified mathematical operations:
1. Quaternion (4D Hyperspheric Rotation)
2. Quaternion operations (conjugate, multiply, angle, slerp)
3. SovereignVector wrapper (PyTorch/NumPy hybrid)
"""

import math
import numpy as np
from typing import Tuple

try:
    import torch
except ImportError:
    torch = None

class Quaternion:
    """
    4D Quaternion (w, x, y, z) for 3D rotation representation.
    """
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @property
    def elements(self):
        return [self.w, self.x, self.y, self.z]

    def __iter__(self):
        return iter([self.w, self.x, self.y, self.z])

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Quaternion':
        n = self.norm()
        if n == 0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other) -> 'Quaternion':
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        
        # Hamilton product
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    def dot(self, other: 'Quaternion') -> float:
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    @property
    def axis(self) -> np.ndarray:
        """Returns the 3D rotation axis vector."""
        q_norm = self.normalize()
        sin_half_theta = math.sqrt(max(0.0, 1.0 - q_norm.w**2))
        if sin_half_theta < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return np.array([q_norm.x / sin_half_theta, q_norm.y / sin_half_theta, q_norm.z / sin_half_theta])

    @property
    def angle(self) -> float:
        """Returns the rotation angle theta in radians."""
        q_norm = self.normalize()
        # Clip to avoid floating point errors out of [-1, 1] range for acos
        w_clipped = min(1.0, max(-1.0, q_norm.w))
        return 2.0 * math.acos(w_clipped)

    @property
    def inverse(self) -> 'Quaternion':
        n_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n_sq == 0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        conj = self.conjugate()
        return Quaternion(conj.w / n_sq, conj.x / n_sq, conj.y / n_sq, conj.z / n_sq)

    @staticmethod
    def distance(q1: 'Quaternion', q2: 'Quaternion') -> float:
        """Angle difference between two unit quaternions."""
        # theta = 2 * acos(|q1 . q2|)
        dot_product = abs(q1.normalize().dot(q2.normalize()))
        dot_product = min(1.0, max(-1.0, dot_product))
        return 2.0 * math.acos(dot_product)

    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', amount: float) -> 'Quaternion':
        """
        Spherical Linear Interpolation (SLERP) between two unit quaternions.
        """
        q1_n = q1.normalize()
        q2_n = q2.normalize()

        dot = q1_n.dot(q2_n)

        # If dot product is negative, reverse direction to take shortest path
        if dot < 0.0:
            q2_n = Quaternion(-q2_n.w, -q2_n.x, -q2_n.y, -q2_n.z)
            dot = -dot

        # If quaternions are very close, use linear interpolation to avoid division by zero
        if dot > 0.9995:
            result = q1_n + (q2_n - q1_n) * amount
            return result.normalize()

        theta_0 = math.acos(dot)
        theta = theta_0 * amount

        sin_theta_0 = math.sin(theta_0)
        sin_theta = math.sin(theta)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion(
            s0 * q1_n.w + s1 * q2_n.w,
            s0 * q1_n.x + s1 * q2_n.x,
            s0 * q1_n.y + s1 * q2_n.y,
            s0 * q1_n.z + s1 * q2_n.z
        ).normalize()

    def __repr__(self):
        return f"Q({self.w:.4f}, {self.x:.4f}i, {self.y:.4f}j, {self.z:.4f}k)"

def traverse_causal_trajectory(content: bytes) -> Quaternion:
    """
    [Phase 51] 위상 기하학의 생명적 창발 (Biological Genesis of Topology)
    통계적 뭉개기를 폐기하고, 바이트 하나하나(세포)가 유입될 때마다
    공간을 비틀어 연속된 인과 궤적(나선)을 그립니다.
    동일한 문자라도 순서(인과)가 다르면 완전히 다른 위치(의미)에 도달합니다.
    """
    if not content:
        return Quaternion(1, 0, 0, 0)
        
    q_current = Quaternion(1.0, 0.0, 0.0, 0.0)
    
    for i, b in enumerate(content):
        angle = (b / 255.0) * math.pi
        
        # 인과(순서)가 영향을 미치도록 회전축을 동적으로 변경
        axis_x = math.sin(i * 0.1)
        axis_y = math.cos(i * 0.1)
        axis_z = math.sin(b * 0.1)
        
        norm = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
        if norm == 0:
            continue
        axis_x /= norm
        axis_y /= norm
        axis_z /= norm
        
        q_cell = Quaternion(
            math.cos(angle / 2.0),
            axis_x * math.sin(angle / 2.0),
            axis_y * math.sin(angle / 2.0),
            axis_z * math.sin(angle / 2.0)
        )
        
        q_current = q_current * q_cell
        
    return q_current.normalize()


_BLADE_MUL_CACHE = {}

class Multivector:
    """
    Multivector in a Clifford/Geometric Algebra Cl(p, q).
    Basis blades are represented by integer bitmasks.
    For example, e_1 -> 1 (001), e_2 -> 2 (010), e_12 -> 3 (011).
    """
    def __init__(self, data: dict, signature: tuple = (3, 0)):
        self.p, self.q = signature
        self.n = self.p + self.q
        # Filter out near-zero elements
        self.data = {int(k): float(v) for k, v in data.items() if abs(v) > 1e-9}

    def _multiply_blades(self, mask1: int, mask2: int) -> tuple:
        """
        Computes the product of two basis blades: mask1 * mask2.
        Returns (result_mask, sign).
        """
        cache_key = (mask1, mask2, self.p, self.q)
        if cache_key in _BLADE_MUL_CACHE:
            return _BLADE_MUL_CACHE[cache_key]

        # 1. Swap count for sorting the index sequence of active dimensions.
        # Compare each bit in mask1 to each bit in mask2.
        # A swap occurs if a bit in mask1 has a higher index than a bit in mask2.
        swaps = 0
        m1 = mask1
        while m1 > 0:
            i = (m1 & -m1).bit_length() - 1
            swaps += bin(mask2 & ((1 << i) - 1)).count('1')
            m1 &= m1 - 1
            
        sign = -1.0 if (swaps % 2) else 1.0

        # 2. Signature squares for overlapping dimensions
        overlap = mask1 & mask2
        o = overlap
        while o > 0:
            idx = (o & -o).bit_length() - 1
            if idx >= self.p:
                sign *= -1.0
            o &= o - 1

        result = mask1 ^ mask2, sign
        _BLADE_MUL_CACHE[cache_key] = result
        return result

    def __add__(self, other: 'Multivector') -> 'Multivector':
        res = self.data.copy()
        for k, v in other.data.items():
            res[k] = res.get(k, 0.0) + v
        return Multivector(res, (self.p, self.q))

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        res = self.data.copy()
        for k, v in other.data.items():
            res[k] = res.get(k, 0.0) - v
        return Multivector(res, (self.p, self.q))

    def __mul__(self, other) -> 'Multivector':
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.data.items()}, (self.p, self.q))
        
        # Geometric product
        res = {}
        for m1, c1 in self.data.items():
            for m2, c2 in other.data.items():
                m3, sign = self._multiply_blades(m1, m2)
                res[m3] = res.get(m3, 0.0) + c1 * c2 * sign
        return Multivector(res, (self.p, self.q))

    def __rmul__(self, other) -> 'Multivector':
        return self.__mul__(other)

    def __xor__(self, other: 'Multivector') -> 'Multivector':
        """
        Outer Product (Wedge Product, ^).
        Only keeps terms where the basis blades do not overlap.
        """
        res = {}
        for m1, c1 in self.data.items():
            for m2, c2 in other.data.items():
                if (m1 & m2) == 0:  # No overlapping dimensions
                    m3, sign = self._multiply_blades(m1, m2)
                    res[m3] = res.get(m3, 0.0) + c1 * c2 * sign
        return Multivector(res, (self.p, self.q))

    def geometric_sync(self, other: 'Multivector') -> Tuple[float, 'Multivector']:
        """
        기하곱(Geometric Product)을 통해 내적(Coherence, 스칼라)과 
        쐐기곱(Phase Torque Bivector, 직교 위상)을 한 번의 연산으로 병렬 추출합니다.
        기존의 .dot()과 ^ 연산이 유발하던 병목과 중복 연산을 제거합니다.
        """
        coherence = 0.0
        wedge_res = {}
        for m1, c1 in self.data.items():
            for m2, c2 in other.data.items():
                if m1 == m2:
                    # 완벽히 같은 차원(동기화된 위상) -> 내적(스칼라)
                    _, sign = self._multiply_blades(m1, m2)
                    coherence += c1 * c2 * sign
                elif (m1 & m2) == 0:
                    # 전혀 겹치지 않는 차원(직교 위상) -> 쐐기곱(새로운 평면 토크)
                    m3, sign = self._multiply_blades(m1, m2)
                    wedge_res[m3] = wedge_res.get(m3, 0.0) + c1 * c2 * sign
        return coherence, Multivector(wedge_res, (self.p, self.q))

    def grade_project(self, grade: int) -> 'Multivector':
        """Returns projection onto components of a specific grade."""
        res = {k: v for k, v in self.data.items() if bin(k).count('1') == grade}
        return Multivector(res, (self.p, self.q))

    def dot(self, other: 'Multivector') -> 'Multivector':
        """
        Inner Product (symmetric dot product / contraction).
        Defined as grade_project(|r - s|) summation over homogeneous parts.
        """
        res = Multivector({}, (self.p, self.q))
        all_keys = list(self.data.keys()) + list(other.data.keys())
        max_grade = max((bin(k).count('1') for k in all_keys), default=0)
        
        for r in range(max_grade + 1):
            a_r = self.grade_project(r)
            if not a_r.data:
                continue
            for s in range(max_grade + 1):
                b_s = other.grade_project(s)
                if not b_s.data:
                    continue
                prod = a_r * b_s
                res = res + prod.grade_project(abs(r - s))
        return res

    def conjugate(self) -> 'Multivector':
        """
        Clifford Reversion.
        Reverses the order of vectors in each blade.
        Sign change factor is (-1)^(g*(g-1)/2) where g is the grade.
        """
        res = {}
        for m, c in self.data.items():
            g = bin(m).count('1')
            factor = -1.0 if ((g * (g - 1) // 2) % 2) else 1.0
            res[m] = c * factor
        return Multivector(res, (self.p, self.q))

    def reverse(self) -> 'Multivector':
        """Alias for conjugate which implements Reversion."""
        return self.conjugate()

    def inverse(self) -> 'Multivector':
        """
        Simple algebraic inverse.
        inverse(A) = A_rev / (A * A_rev) if A * A_rev is purely scalar.
        """
        rev = self.conjugate()
        norm_sq_mv = self * rev
        scalar = norm_sq_mv.data.get(0, 0.0)
        if len(norm_sq_mv.data) > 1 or abs(scalar) < 1e-9:
            raise ValueError("This multivector does not have a simple scalar norm square and cannot be inverted.")
        return rev * (1.0 / scalar)

    def dual(self) -> 'Multivector':
        """
        Hodge Dual (Geometric Algebra Dual).
        A* = A * I^{-1}, where I is the pseudoscalar of the space.
        Represents the 'Negative Space' (Void) of the multivector (Wave).
        """
        I_mask = (1 << self.n) - 1
        I_mv = Multivector({I_mask: 1.0}, (self.p, self.q))
        try:
            I_inv = I_mv.inverse()
        except ValueError:
            # Fallback if inverse fails for degenerate signatures
            I_inv = Multivector({I_mask: -1.0}, (self.p, self.q))
        return self * I_inv

    def delta_coupling(self, other1: 'Multivector', other2: 'Multivector') -> 'Multivector':
        """
        Delta (Δ) Coupling: 순환 차이 연산
        세 멀티벡터(또는 바이벡터) B1(self), B2(other1), B3(other2) 간의 차이가
        순환 고리를 이루도록 계산하여 순환 텐션(잔여 오차)을 반환합니다.
        (B1 - B2) + (B2 - B3) + (B3 - B1) 은 수학적으로 0이지만,
        여기서는 각각의 텐션의 노름(norm)을 성분별로 누적하여 순환 고리 내부의
        잔여 텐션의 절대적인 크기와 방향을 표현하는 새로운 멀티벡터를 반환합니다.
        """
        diff1 = self - other1
        diff2 = other1 - other2
        diff3 = other2 - self

        # 순환 고리에서의 각 성분별 에너지(절대 차이의 합)를 계산하여 반환
        res = {}
        for m in set(diff1.data.keys()) | set(diff2.data.keys()) | set(diff3.data.keys()):
            val = abs(diff1.data.get(m, 0.0)) + abs(diff2.data.get(m, 0.0)) + abs(diff3.data.get(m, 0.0))
            if val > 1e-9:
                res[m] = val
        return Multivector(res, (self.p, self.q))

    def wye_synchronize(self, other1: 'Multivector', other2: 'Multivector') -> 'Multivector':
        """
        Wye (Y) Synchronization: 중성점 수렴 연산
        세 멀티벡터(또는 바이벡터) B1(self), B2(other1), B3(other2)의
        합성 영점(Neutral Point, 중성점)으로 수렴하는 텐션을 반환합니다.
        (B1 + B2 + B3) / 3 을 계산하여 완전한 삼대칭(120도 동기화) 상태의 위상 중심을 도출합니다.
        """
        sum_mv = self + other1 + other2
        return sum_mv * (1.0 / 3.0)

    def __repr__(self):
        if not self.data:
            return "0"
        terms = []
        sorted_keys = sorted(self.data.keys(), key=lambda k: (bin(k).count('1'), k))
        for k in sorted_keys:
            v = self.data[k]
            if k == 0:
                terms.append(f"{v:.4f}")
            else:
                indices = []
                temp = k
                idx = 1
                while temp > 0:
                    if temp & 1:
                        indices.append(str(idx))
                    temp >>= 1
                    idx += 1
                blade_str = "e" + "".join(indices)
                terms.append(f"{v:.4f}*{blade_str}")
        return " + ".join(terms).replace("+ -", "- ")

class ConformalSpace:
    """
    등각 기하대수(CGA, Conformal Geometric Algebra) Cl(4,1) 구현체.
    유클리드 공간을 한 차원 높여서 평행이동과 팽창을 회전(Motor)으로 통일합니다.
    - 차원 0, 1, 2 : e1, e2, e3 (+1 signature)
    - 차원 3 : e+ (+1 signature)
    - 차원 4 : e- (-1 signature)
    """
    SIGNATURE = (4, 1)

    # 기본 기저 벡터
    e1 = Multivector({1: 1.0}, SIGNATURE)
    e2 = Multivector({2: 1.0}, SIGNATURE)
    e3 = Multivector({4: 1.0}, SIGNATURE)
    e_plus = Multivector({8: 1.0}, SIGNATURE)
    e_minus = Multivector({16: 1.0}, SIGNATURE)

    # Null Basis (원점과 무한대)
    # eo = 0.5 * (e_minus - e_plus)
    eo = Multivector({8: -0.5, 16: 0.5}, SIGNATURE)
    # einf = e_minus + e_plus
    einf = Multivector({8: 1.0, 16: 1.0}, SIGNATURE)

    # Minkowski 평면 블레이드 (E = einf ^ eo = e_plus ^ e_minus)
    # e_plus * e_minus => mask 24 (8 ^ 16)
    E = Multivector({24: 1.0}, SIGNATURE)

    @classmethod
    def up(cls, x: float, y: float, z: float) -> Multivector:
        """유클리드 좌표를 등각 공간(Null Vector)으로 맵핑합니다."""
        v = x * cls.e1 + y * cls.e2 + z * cls.e3
        v2 = x**2 + y**2 + z**2
        # X = v + 0.5 * v^2 * einf + eo
        return v + cls.einf * (0.5 * v2) + cls.eo

    @classmethod
    def down(cls, X: Multivector) -> Tuple[float, float, float]:
        """등각 공간의 Null Vector를 다시 유클리드 좌표로 투영합니다."""
        # -X / (X \cdot einf)
        # 내적(dot)은 구현된 grade_project summation을 사용
        X_dot_einf = X.dot(cls.einf).data.get(0, -1.0)
        if abs(X_dot_einf) < 1e-9:
            return (0.0, 0.0, 0.0) # 무한대 포인트 보호
        
        proj = X * (-1.0 / X_dot_einf)
        return (proj.data.get(1, 0.0), proj.data.get(2, 0.0), proj.data.get(4, 0.0))

    @classmethod
    def translator(cls, tx: float, ty: float, tz: float) -> Multivector:
        """평행 이동을 생성하는 모터(Translator)를 반환합니다."""
        t_vec = tx * cls.e1 + ty * cls.e2 + tz * cls.e3
        # T = 1 - 0.5 * t_vec * einf
        return Multivector({0: 1.0}, cls.SIGNATURE) - (t_vec * cls.einf) * 0.5

    @classmethod
    def dilator(cls, scale: float) -> Multivector:
        """공간 팽창/수축을 생성하는 모터(Dilator)를 반환합니다."""
        if scale <= 0: scale = 1e-9
        gamma = math.log(scale)
        # D = exp(-gamma/2 * E) = cosh(gamma/2) - sinh(gamma/2) * E
        ch = math.cosh(gamma / 2.0)
        sh = math.sinh(-gamma / 2.0)
        return Multivector({0: ch}, cls.SIGNATURE) + cls.E * sh

    @classmethod
    def apply_motor(cls, motor: Multivector, X: Multivector) -> Multivector:
        """Motor(Translator, Dilator, Rotor)를 스핀 샌드위치 연산으로 적용합니다: M * X * ~M"""
        reverse_motor = motor.reverse()
        return motor * X * reverse_motor

class RotorOperator:
    """
    [Roadmap: Meta-Rotorization]
    Rotor Operator that acts on a Constant Axis (Invariant Core) 
    via a Spin Sandwich product (R * C * ~R) to represent a Covariant Layer.
    """
    def __init__(self, operator):
        self.operator = operator # Can be a Quaternion or Multivector

    def act_on(self, constant_axis):
        if isinstance(self.operator, Quaternion) and isinstance(constant_axis, Quaternion):
            return (self.operator * constant_axis * self.operator.conjugate()).normalize()
        elif isinstance(self.operator, Multivector) and isinstance(constant_axis, Multivector):
            return self.operator * constant_axis * self.operator.reverse()
        else:
            raise TypeError("Mismatch of operator and constant axis types. Both must be Quaternion or both Multivector.")
