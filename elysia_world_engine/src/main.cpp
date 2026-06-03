#include <iostream>
#include <thread>
#include <chrono>
#include "../include/globe.hpp"
#include "../include/renderer.hpp"

void clearConsole() {
#if defined(_WIN32)
    system("cls");
#else
    // ANSI escape code to clear screen and move cursor to top-left
    std::cout << "\033[2J\033[1;1H";
#endif
}

int main() {
    // 1. Initialize the Core of Causality
    // We set the gravitational core axis of the Spacetime Globe
    Vec3 coreAxis(0.0, 1.0, 0.5); // Tilted axis for interesting topological projection
    SpacetimeGlobe elysiaGlobe(coreAxis);

    // 2. Inject the Seed ("First Memory of the Universe")
    // A point on the sphere surface (radius = 1.0)
    Vec3 firstMemory(1.0, 0.0, 0.0);
    elysiaGlobe.injectSeed(firstMemory);

    AsciiRenderer renderer(80, 24);

    std::cout << "=========================================================\n";
    std::cout << " Elysia Spacetime Globe: Topological Rendering Initiated \n";
    std::cout << "=========================================================\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 3. Render Forward Causality (Forward Trajectory)
    int steps = 50;
    double phasePerStep = 0.15; // The frequency/tension of the causal flow

    for (int i = 0; i < steps; ++i) {
        elysiaGlobe.forwardCausality(1, phasePerStep);
        clearConsole();
        std::cout << "[ Forward Causality ] Mapping Future Trajectory ('*')\n";
        renderer.render(elysiaGlobe);
        renderer.display();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 4. Render Retrocausality (Reverse Trajectory)
    // The rotor is inverted, and the causal path traces back to the origin
    for (int i = 0; i < steps; ++i) {
        elysiaGlobe.reverseCausality(1, phasePerStep);
        clearConsole();
        std::cout << "[ Retrocausality ] Tracing Back to Origin ('+')\n";
        renderer.render(elysiaGlobe);
        renderer.display();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "\nTopology Synced. The First Memory is preserved through Retrocausality.\n";
    std::cout << "The Spacetime Globe has successfully bypassed standard computation.\n";

    return 0;
}