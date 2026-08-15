#ifndef CAUSAL_MAIN_LOOP_H
#define CAUSAL_MAIN_LOOP_H

#include "core/os/main_loop.h"
#include "causal_server.h"
#include "causal_server_simd.h"

class CausalMainLoop : public MainLoop {
    GDCLASS(CausalMainLoop, MainLoop);

public:
    virtual void initialize() override {
    }

    virtual bool iteration(double p_delta) override {
        if (CausalServerSIMD::get_singleton()) {
            CausalServerSIMD::get_singleton()->propagate_and_project_parallel();
        } else if (CausalServer::get_singleton()) {
            CausalServer::get_singleton()->propagate_and_project();
        }

        return false;
    }

    virtual void finalize() override {
    }
};

#endif // CAUSAL_MAIN_LOOP_H
