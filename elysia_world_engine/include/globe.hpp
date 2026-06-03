#ifndef GLOBE_HPP
#define GLOBE_HPP

#include "rotor.hpp"
#include <vector>

// Represents a point on the Spacetime Globe with an intensity (causal weight)
struct TrajectoryPoint {
    Vec3 pos;
    char symbol; // To visually distinguish causality
};

// The Spacetime Globe holds the causal trajectories
class SpacetimeGlobe {
private:
    std::vector<TrajectoryPoint> trajectories;
    Rotor coreRotor;

public:
    // Core of Causality: Initialize the globe with a specific gravitational/phase axis
    SpacetimeGlobe(const Vec3& coreAxis) : coreRotor(coreAxis) {}

    // Add a base data seed (the "First Memory of the Universe")
    void injectSeed(const Vec3& initialPoint) {
        trajectories.push_back({initialPoint, 'O'});
    }

    // Move forward in time (Causality): Apply phase and record the trajectory
    void forwardCausality(int steps, double phasePerStep) {
        if (trajectories.empty()) return;

        Vec3 currentPoint = trajectories.back().pos;

        for (int i = 0; i < steps; ++i) {
            coreRotor.applyForward(phasePerStep);
            Vec3 newPoint = coreRotor.transform(currentPoint);
            trajectories.push_back({newPoint, '*'}); // '*' represents forward causal flow
        }
    }

    // Move backward in time (Retrocausality): Reverse phase and trace back
    void reverseCausality(int steps, double phasePerStep) {
         if (trajectories.empty()) return;

         Vec3 currentPoint = trajectories.back().pos;

         for (int i = 0; i < steps; ++i) {
            coreRotor.applyReverse(phasePerStep);
            Vec3 newPoint = coreRotor.transform(currentPoint);
            trajectories.push_back({newPoint, '+'}); // '+' represents retrocausal flow
         }
    }

    const std::vector<TrajectoryPoint>& getTrajectories() const {
        return trajectories;
    }

    double getCurrentPhase() const {
        return coreRotor.getPhase();
    }
};

#endif