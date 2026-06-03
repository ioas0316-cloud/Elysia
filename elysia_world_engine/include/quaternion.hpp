#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <cmath>

struct Vec3 {
    double x, y, z;

    Vec3(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
};

class Quaternion {
public:
    double w, x, y, z;

    Quaternion(double w=1, double x=0, double y=0, double z=0) : w(w), x(x), y(y), z(z) {}

    // Create a quaternion from an axis and an angle (in radians)
    static Quaternion fromAxisAngle(const Vec3& axis, double angle) {
        double halfAngle = angle * 0.5;
        double s = std::sin(halfAngle);
        // Normalize axis
        double len = std::sqrt(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
        if (len == 0.0) return Quaternion();

        return Quaternion(
            std::cos(halfAngle),
            (axis.x / len) * s,
            (axis.y / len) * s,
            (axis.z / len) * s
        );
    }

    // Quaternion multiplication
    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        );
    }

    // Conjugate (Inverse for unit quaternions)
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    // Rotate a 3D vector using this quaternion
    Vec3 rotate(const Vec3& v) const {
        Quaternion q_v(0, v.x, v.y, v.z);
        Quaternion q_res = (*this) * q_v * this->conjugate();
        return Vec3(q_res.x, q_res.y, q_res.z);
    }
};

#endif
