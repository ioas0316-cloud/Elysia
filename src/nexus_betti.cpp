#include "nexus_betti.h"

namespace CausalNexus {

BettiNumbers CalculateBetti2D(const std::vector<uint8_t>& grid, int H, int W) {
    if (grid.empty() || H <= 0 || W <= 0 || static_cast<int>(grid.size()) < H * W) {
        return {0, 0};
    }

    int total_pixels = H * W;
    DisjointSet dsu(total_pixels);

    int V = 0; // Active pixel vertices
    int E = 0; // 4-connected edges
    int F = 0; // 2x2 occupied pixel faces

    auto get_idx = [W](int r, int c) { return r * W + c; };

    // 1. Calculate Vertices, Edges, and build DSU for connected components
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (!grid[get_idx(r, c)]) continue;
            V++;

            // Right neighbor
            if (c + 1 < W && grid[get_idx(r, c + 1)]) {
                E++;
                dsu.unite(get_idx(r, c), get_idx(r, c + 1));
            }

            // Bottom neighbor
            if (r + 1 < H && grid[get_idx(r + 1, c)]) {
                E++;
                dsu.unite(get_idx(r, c), get_idx(r + 1, c));
            }
        }
    }

    // 2. Calculate Faces (2x2 filled pixel quads)
    for (int r = 0; r < H - 1; ++r) {
        for (int c = 0; c < W - 1; ++c) {
            if (grid[get_idx(r, c)] && grid[get_idx(r, c + 1)] &&
                grid[get_idx(r + 1, c)] && grid[get_idx(r + 1, c + 1)]) {
                F++;
            }
        }
    }

    // 3. Betti 0: Number of distinct DSU root nodes
    int betti_0 = 0;
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int idx = get_idx(r, c);
            if (grid[idx] && dsu.find(idx) == idx) {
                betti_0++;
            }
        }
    }

    // 4. Euler Characteristic: chi = V - E + F = betti_0 - betti_1
    int chi = V - E + F;
    int betti_1 = betti_0 - chi;

    return {betti_0, std::max(0, betti_1)};
}

} // namespace CausalNexus
