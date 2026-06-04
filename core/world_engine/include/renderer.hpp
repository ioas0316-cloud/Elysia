#ifndef RENDERER_HPP
#define RENDERER_HPP

#include "globe.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

class AsciiRenderer {
private:
    int width, height;
    std::vector<std::vector<char>> buffer;

    // Clear the rendering buffer
    void clear() {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer[y][x] = ' ';
            }
        }
    }

public:
    AsciiRenderer(int w, int h) : width(w), height(h) {
        buffer.resize(height, std::vector<char>(width, ' '));
    }

    // Render the Spacetime Globe's trajectories into the ASCII buffer
    void render(const SpacetimeGlobe& globe) {
        clear();

        // Draw a basic sphere boundary for context
        double radius = std::min(width, height) / 2.5;
        double cx = width / 2.0;
        double cy = height / 2.0;

        for(double t = 0; t < 2 * M_PI; t += 0.1) {
             int x = static_cast<int>(cx + radius * std::cos(t) * 2.0); // Aspect ratio correction
             int y = static_cast<int>(cy + radius * std::sin(t));
             if (x >= 0 && x < width && y >= 0 && y < height) {
                 buffer[y][x] = '.';
             }
        }

        const auto& trajectories = globe.getTrajectories();
        for (const auto& tp : trajectories) {
            // Project 3D vector (x,y,z) onto 2D ASCII plane
            // Assuming the vector magnitudes are normalized around 1.0

            // Map [-1, 1] to screen space
            int screenX = static_cast<int>(cx + (tp.pos.x * radius * 2.0)); // x2 for console aspect ratio
            int screenY = static_cast<int>(cy - (tp.pos.y * radius));      // Invert Y for console

            if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
                buffer[screenY][screenX] = tp.symbol;
            }
        }
    }

    // Print the buffer to console
    void display() const {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << buffer[y][x];
            }
            std::cout << '\n';
        }
    }
};

#endif