import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.math_utils import ConformalSpace, Multivector

class TestCGAMotors(unittest.TestCase):
    def test_up_down_mapping(self):
        """유클리드 공간 -> 등각 공간 -> 유클리드 공간 변환 무결성 테스트"""
        x, y, z = 3.0, 4.0, 5.0
        X = ConformalSpace.up(x, y, z)
        
        # Null vector check: X^2 == 0
        X_sq = X * X
        self.assertAlmostEqual(X_sq.data.get(0, 0.0), 0.0, places=5)
        
        # Down mapping
        rx, ry, rz = ConformalSpace.down(X)
        self.assertAlmostEqual(rx, x, places=5)
        self.assertAlmostEqual(ry, y, places=5)
        self.assertAlmostEqual(rz, z, places=5)

    def test_translator_motor(self):
        """평행이동 모터 테스트 (Translation as Rotation)"""
        x, y, z = 1.0, 2.0, 3.0
        tx, ty, tz = 5.0, -1.0, 2.0
        
        X = ConformalSpace.up(x, y, z)
        T = ConformalSpace.translator(tx, ty, tz)
        
        # Apply Motor
        X_moved = ConformalSpace.apply_motor(T, X)
        
        # Verify
        rx, ry, rz = ConformalSpace.down(X_moved)
        self.assertAlmostEqual(rx, x + tx, places=5)
        self.assertAlmostEqual(ry, y + ty, places=5)
        self.assertAlmostEqual(rz, z + tz, places=5)

    def test_dilator_motor(self):
        """팽창 수축 모터 테스트 (Dilation as Rotation)"""
        x, y, z = 2.0, 0.0, 0.0
        scale = 3.5
        
        X = ConformalSpace.up(x, y, z)
        D = ConformalSpace.dilator(scale)
        
        # Apply Motor
        X_scaled = ConformalSpace.apply_motor(D, X)
        
        # Verify
        rx, ry, rz = ConformalSpace.down(X_scaled)
        self.assertAlmostEqual(rx, x * scale, places=5)
        self.assertAlmostEqual(ry, y * scale, places=5)
        self.assertAlmostEqual(rz, z * scale, places=5)

    def test_compound_motor(self):
        """스케일과 트랜스레이션 복합 기어 작용 테스트"""
        x, y, z = 1.0, 0.0, 0.0
        scale = 2.0
        tx, ty, tz = 0.0, 5.0, 0.0
        
        X = ConformalSpace.up(x, y, z)
        
        # Scale first, then Translate
        D = ConformalSpace.dilator(scale)
        T = ConformalSpace.translator(tx, ty, tz)
        
        M = T * D  # 모터는 기하곱으로 합성됨!
        
        X_final = ConformalSpace.apply_motor(M, X)
        rx, ry, rz = ConformalSpace.down(X_final)
        
        self.assertAlmostEqual(rx, x * scale + tx, places=5)
        self.assertAlmostEqual(ry, y * scale + ty, places=5)
        self.assertAlmostEqual(rz, z * scale + tz, places=5)

if __name__ == "__main__":
    unittest.main()
