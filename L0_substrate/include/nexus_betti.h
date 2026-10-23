#ifndef NEXUS_BETTI_H
#define NEXUS_BETTI_H

#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace CausalNexus {

struct BettiNumbers {
    int betti_0; // Connected components count
    int betti_1; // Holes / Cycles count
};

class DisjointSet {
public:
    std::vector<int> parent;
    explicit DisjointSet(int n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }

    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
        }
    }
};

/**
 * Calculates 2D Betti numbers (betti_0 and betti_1) over a binary uint8_t grid mask.
 * Uses Euler Characteristic: chi = V - E + F = betti_0 - betti_1
 * Complexity: O(H * W)
 */
BettiNumbers CalculateBetti2D(const std::vector<uint8_t>& grid, int H, int W);

} // namespace CausalNexus

#endif // NEXUS_BETTI_H
