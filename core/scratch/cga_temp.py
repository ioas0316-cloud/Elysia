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
        # D = cosh(gamma/2) + sinh(gamma/2) * E
        ch = math.cosh(gamma / 2.0)
        sh = math.sinh(gamma / 2.0)
        return Multivector({0: ch}, cls.SIGNATURE) + cls.E * sh

    @classmethod
    def apply_motor(cls, motor: Multivector, X: Multivector) -> Multivector:
        """Motor(Translator, Dilator, Rotor)를 스핀 샌드위치 연산으로 적용합니다: M * X * ~M"""
        return motor * X * motor.conjugate()
