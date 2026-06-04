#ifndef ROTOR_HPP
#define ROTOR_HPP

#include "quaternion.hpp"

// The Variable Rotor represents the core axis of a Spacetime Globe.
// It applies topological phase tension to input data (causes)
// and can reverse the process (retrocausality).
class Rotor {
private:
    Vec3 axis;
    double currentPhase;
    Quaternion currentRotation;

public:
    Rotor(const Vec3& coreAxis, double initialPhase = 0.0)
        : axis(coreAxis), currentPhase(initialPhase) {
        currentRotation = Quaternion::fromAxisAngle(axis, currentPhase);
    }

    // Apply forward causality (adds to phase angle)
    void applyForward(double phaseDelta) {
        currentPhase += phaseDelta;
        currentRotation = Quaternion::fromAxisAngle(axis, currentPhase);
    }

    // Apply retrocausality (subtracts from phase angle, effectively reversing time/causality)
    void applyReverse(double phaseDelta) {
        currentPhase -= phaseDelta;
        currentRotation = Quaternion::fromAxisAngle(axis, currentPhase);
    }

    // Rotate a data point (vector) in the 3D space according to the current tension
    Vec3 transform(const Vec3& point) const {
        return currentRotation.rotate(point);
    }

    // Invert the entire rotor state (swap forward/reverse trajectories completely)
    void invert() {
        currentPhase = -currentPhase;
        currentRotation = Quaternion::fromAxisAngle(axis, currentPhase);
    }

    double getPhase() const { return currentPhase; }
};

#endif