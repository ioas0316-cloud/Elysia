#ifndef GEOMETRIC_ISA_HPP
#define GEOMETRIC_ISA_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace elysia::isa {

// ============================================================================
// 1. Geometric Algebra Instruction Set Architecture (ISA) Definitions
// ============================================================================

enum class Opcode : uint8_t {
    PLOCK      = 0x01,  ///< Spinner Phase Lock (스피너 위상 고정)
    ROTOR_PIN  = 0x02,  ///< Pin Delta Rotor to SRAM-R (로터 핀 저장)
    GEOM_SWAP  = 0x03,  ///< Geometric Tier Swap with Phase Recovery
    METRIC_MOD = 0x04,  ///< Modify g_mem tensor metric field
    CHART_SUB  = 0x05,  ///< Octree Chart Subdivide
    CHART_MRG  = 0x06   ///< Octree Chart Merge
};

struct Instruction {
    Opcode opcode;
    uint32_t src_reg;
    uint32_t dst_reg;
    uint64_t memory_address;
    float immediate_param;

    std::string to_assembly() const {
        std::ostringstream ss;
        switch (opcode) {
            case Opcode::PLOCK:
                ss << "PLOCK R" << dst_reg << ", 0x" << std::hex << memory_address;
                break;
            case Opcode::ROTOR_PIN:
                ss << "ROTOR_PIN R" << src_reg << ", SRAM_R[0x" << std::hex << memory_address << "]";
                break;
            case Opcode::GEOM_SWAP:
                ss << "GEOM_SWAP R" << src_reg << ", R" << dst_reg << ", 0x" << std::hex << memory_address;
                break;
            case Opcode::METRIC_MOD:
                ss << "METRIC_MOD R" << dst_reg << ", param=" << std::fixed << std::setprecision(3) << immediate_param;
                break;
            case Opcode::CHART_SUB:
                ss << "CHART_SUB ChartID=" << src_reg;
                break;
            case Opcode::CHART_MRG:
                ss << "CHART_MRG ChartID=" << src_reg;
                break;
        }
        return ss.str();
    }
};

// ============================================================================
// 2. Phase-Aware Compiler Pass
// ============================================================================

class PhaseAwareCompilerPass {
public:
    PhaseAwareCompilerPass() = default;

    struct ProgramMemoryAccess {
        uint64_t addr;
        bool is_swap_boundary;
        bool causes_eviction;
        float phase_drift;
    };

    std::vector<Instruction> transform_and_emit(const std::vector<ProgramMemoryAccess>& accesses) {
        std::vector<Instruction> isa_stream;

        for (size_t i = 0; i < accesses.size(); ++i) {
            const auto& access = accesses[i];

            if (access.causes_eviction) {
                // Insert ROTOR_PIN instruction before eviction
                Instruction pin_inst{};
                pin_inst.opcode = Opcode::ROTOR_PIN;
                pin_inst.src_reg = 1; // Current active rotor reg
                pin_inst.memory_address = access.addr;
                isa_stream.push_back(pin_inst);
            }

            if (access.is_swap_boundary) {
                // Insert GEOM_SWAP and PLOCK instructions
                Instruction swap_inst{};
                swap_inst.opcode = Opcode::GEOM_SWAP;
                swap_inst.src_reg = 1;
                swap_inst.dst_reg = 2;
                swap_inst.memory_address = access.addr;
                isa_stream.push_back(swap_inst);

                Instruction lock_inst{};
                lock_inst.opcode = Opcode::PLOCK;
                lock_inst.dst_reg = 2;
                lock_inst.memory_address = access.addr;
                lock_inst.immediate_param = access.phase_drift;
                isa_stream.push_back(lock_inst);
            }
        }

        return isa_stream;
    }
};

} // namespace elysia::isa

#endif // GEOMETRIC_ISA_HPP
