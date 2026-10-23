#ifndef CAUSAL_EDITOR_PLUGIN_H
#define CAUSAL_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "editor/editor_interface.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "causal_graph_control.h"

struct BenchmarkResult;
extern BenchmarkResult run_causal_simd_100k_benchmark_with_data();

class CausalEditorPlugin : public EditorPlugin {
    GDCLASS(CausalEditorPlugin, EditorPlugin);

private:
    AcceptDialog *bench_dialog = nullptr;
    CausalGraphControl *graph_control = nullptr;
    Label *summary_label = nullptr;

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("_on_benchmark_menu_pressed"), &CausalEditorPlugin::_on_benchmark_pressed);
    }

    void _on_benchmark_pressed() {
        BenchmarkResult res = run_causal_simd_100k_benchmark_with_data();

        if (!bench_dialog) {
            bench_dialog = memnew(AcceptDialog);
            bench_dialog->set_title("Causal SIMD 100k Benchmark Performance Graph");

            VBoxContainer *vbox = memnew(VBoxContainer);
            bench_dialog->add_child(vbox);

            summary_label = memnew(Label);
            vbox->add_child(summary_label);

            graph_control = memnew(CausalGraphControl);
            graph_control->set_custom_minimum_size(Vector2(480, 180));
            vbox->add_child(graph_control);

            if (EditorInterface::get_singleton() && EditorInterface::get_singleton()->get_base_control()) {
                Control *base_control = EditorInterface::get_singleton()->get_base_control();
                base_control->add_child(bench_dialog);
            }
        }

        summary_label->set_text(res.summary_text);
        graph_control->set_data(res.iteration_times_us);

        bench_dialog->popup_centered(Vector2i(520, 280));
    }

public:
    virtual String get_name() const override { return "CausalTopologyTools"; }

    virtual void _enter_tree() override {
        add_tool_menu_item("Run Causal 100k Benchmark", Callable(this, "_on_benchmark_menu_pressed"));
    }

    virtual void _exit_tree() override {
        remove_tool_menu_item("Run Causal 100k Benchmark");

        if (bench_dialog) {
            bench_dialog->queue_free();
            bench_dialog = nullptr;
        }
    }
};

#endif // CAUSAL_EDITOR_PLUGIN_H
