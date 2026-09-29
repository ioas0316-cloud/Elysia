#include "isa_compiler/topir/eol_parser.hpp"
#include "isa_compiler/codegen/codegen_engine.hpp"
#include "include/topir_runtime_engine.hpp"
#include <iostream>
#include <cassert>

int main() {
    using namespace elysia::topir;
    using namespace elysia::topir::codegen;

    std::string eol_code = "pipeline EvolveSystem(dt) { ... }";
    TopologicalGraphIR graph = EOLParser::parse_to_topir(eol_code);

    std::string hlsl_code = CodeGeneratorEngine::compile_to_backend(graph, BackendTarget::HLSL_ComputeShader);
    assert(!hlsl_code.empty());
    std::cout << "[Test TopIR C++ Pipeline] Generated HLSL Shader code successfully (" << hlsl_code.size() << " bytes).\n";

    std::string cpp_code = CodeGeneratorEngine::compile_to_backend(graph, BackendTarget::CPP_CPU);
    assert(!cpp_code.empty());
    std::cout << "[Test TopIR C++ Pipeline] Generated C++ code successfully (" << cpp_code.size() << " bytes).\n";

    elysia::runtime::RuntimeConfig config;
    config.grid_dim = 16;
    config.dt = 0.005f;
    config.K_0 = 10.0f;

    elysia::runtime::TopIRRuntimeEngine runtime(config);
    for (int i = 0; i < 10; ++i) {
        runtime.step();
    }

    const float* q_field = runtime.get_q_field();
    float w_norm = q_field[3];
    std::cout << "[Test TopIR C++ Pipeline] Step completed. Center voxel rotor w: " << w_norm << "\n";
    assert(w_norm > 0.0f && w_norm <= 1.0f);

    std::cout << "[Test TopIR C++ Pipeline] All assertions passed successfully!\n";
    return 0;
}
