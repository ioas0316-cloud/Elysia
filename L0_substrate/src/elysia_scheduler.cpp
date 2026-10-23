#include "elysia_scheduler.hpp"
#include <iostream>

// Helper functions or pipeline utilities for Elysia FOC Scheduler
void print_elysia_foc_status(const VRAMState& vram, float gamma_d, float gaba_th, float ach_level) {
    std::cout << "[Elysia FOC Scheduler] VRAM: " << vram.used_mb << "MB / " << (vram.used_mb + vram.free_mb) << "MB "
              << " | Gamma_D: " << gamma_d
              << " | GABA_Th: " << gaba_th
              << " | ACh_Level: " << ach_level << std::endl;
}
