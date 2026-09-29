#ifndef CODEGEN_ENGINE_HPP
#define CODEGEN_ENGINE_HPP

#include "hlsl_codegen.hpp"
#include "cpp_cuda_codegen.hpp"
#include "../topir/topir_passes.hpp"

namespace elysia::topir::codegen {

enum class BackendTarget {
    HLSL_ComputeShader,
    CPP_CPU,
    CUDA_Vulkan
};

class CodeGeneratorEngine {
public:
    CodeGeneratorEngine() = default;

    static std::string compile_to_backend(TopologicalGraphIR& graph, BackendTarget target) {
        // Run optimization passes before codegen
        ContinuousBranchNeutralizationPass::run(graph);
        AlgebraicIsomorphismReductionPass::run(graph);

        switch (target) {
            case BackendTarget::HLSL_ComputeShader:
                return HLSLCodeGenerator::generate_hlsl(graph);
            case BackendTarget::CPP_CPU:
            case BackendTarget::CUDA_Vulkan:
                return CppCudaCodeGenerator::generate_cpp(graph);
        }
        return "// Unsupported backend target";
    }
};

} // namespace elysia::topir::codegen

#endif // CODEGEN_ENGINE_HPP
