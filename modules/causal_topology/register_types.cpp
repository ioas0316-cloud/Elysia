#include "register_types.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "causal_server.h"
#include "causal_server_simd.h"
#include "causal_main_loop.h"
#include "causal_editor_plugin.h"

static CausalServer *causal_server_ptr = nullptr;
static CausalServerSIMD *causal_server_simd_ptr = nullptr;

extern void run_causal_simd_100k_benchmark();

void initialize_causal_topology_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
        GDREGISTER_CLASS(CausalServer);
        GDREGISTER_CLASS(CausalServerSIMD);
        GDREGISTER_CLASS(CausalMainLoop);

        causal_server_ptr = memnew(CausalServer);
        Engine::get_singleton()->add_singleton(Engine::Singleton("CausalServer", CausalServer::get_singleton()));

        causal_server_simd_ptr = memnew(CausalServerSIMD);
        Engine::get_singleton()->add_singleton(Engine::Singleton("CausalServerSIMD", CausalServerSIMD::get_singleton()));

        List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();
        for (const String &arg : cmdline_args) {
            if (arg == "--run-causal-bench") {
                print_line("[CLI] Triggering Causal SIMD 100k Benchmark...");
                run_causal_simd_100k_benchmark();
                break;
            }
        }
    }

    if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
        GDREGISTER_CLASS(CausalEditorPlugin);
        GDREGISTER_CLASS(CausalGraphControl);
        EditorPlugins::add_by_type<CausalEditorPlugin>();
    }
}

void uninitialize_causal_topology_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
        if (causal_server_simd_ptr) {
            memdelete(causal_server_simd_ptr);
            causal_server_simd_ptr = nullptr;
        }
        if (causal_server_ptr) {
            memdelete(causal_server_ptr);
            causal_server_ptr = nullptr;
        }
    }
}
