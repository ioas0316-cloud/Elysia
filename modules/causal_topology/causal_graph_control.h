#ifndef CAUSAL_GRAPH_CONTROL_H
#define CAUSAL_GRAPH_CONTROL_H

#include "scene/gui/control.h"
#include "scene/theme/theme_db.h"
#include "core/os/worker_thread_pool.h"
#include "core/templates/vector.h"

FORCE_INLINE String _get_simd_architecture_name() {
#if defined(__AVX2__)
    return "AVX2 (256-bit FMA)";
#elif defined(__ARM_NEON)
    return "ARM NEON (128-bit)";
#elif defined(__SSE2__)
    return "SSE2 (128-bit)";
#else
    return "Scalar (Fallback)";
#endif
}

class CausalGraphControl : public Control {
    GDCLASS(CausalGraphControl, Control);

private:
    Vector<double> iteration_times_us;

protected:
    static void _bind_methods() {}

    void _notification(int p_what) {
        if (p_what == NOTIFICATION_DRAW) {
            _draw_graph();
        }
    }

    void _draw_graph() {
        Vector2 size = get_size();
        if (size.x <= 0 || size.y <= 0 || iteration_times_us.is_empty()) return;

        draw_rect(Rect2(Vector2(), size), Color(0.1f, 0.11f, 0.13f, 1.0f));
        draw_rect(Rect2(Vector2(), size), Color(0.3f, 0.35f, 0.4f, 1.0f), false, 1.0f);

        double max_time = 1.0;
        for (int i = 0; i < iteration_times_us.size(); i++) {
            if (iteration_times_us[i] > max_time) max_time = iteration_times_us[i];
        }
        max_time *= 1.25;

        float margin = 25.0f;
        float chart_w = size.x - (margin * 2.0f);
        float chart_h = size.y - (margin * 2.0f);
        int count = iteration_times_us.size();
        float x_step = (count > 1) ? chart_w / (count - 1) : chart_w;

        PackedVector2Array polyline_points;
        Ref<Font> font = ThemeDB::get_singleton() ? ThemeDB::get_singleton()->get_fallback_font() : Ref<Font>();

        for (int i = 0; i < count; i++) {
            float x = margin + (i * x_step);
            float ratio = (float)(iteration_times_us[i] / max_time);
            float y = size.y - margin - (ratio * chart_h);

            Vector2 point(x, y);
            polyline_points.push_back(point);

            draw_circle(point, 3.5f, Color(0.2f, 0.85f, 1.0f, 1.0f));

            if (font.is_valid()) {
                String val_str = vformat("%.0fus", iteration_times_us[i]);
                draw_string(font, Vector2(x - 12.0f, y - 6.0f), val_str, HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.8f, 0.8f, 0.8f));
            }
        }

        if (polyline_points.size() > 1) {
            draw_polyline(polyline_points, Color(0.2f, 0.9f, 0.4f, 1.0f), 2.0f);
        }

        Vector2 legend_size = Vector2(210.0f, 48.0f);
        Vector2 legend_pos = Vector2(size.x - legend_size.x - 10.0f, 10.0f);

        draw_rect(Rect2(legend_pos, legend_size), Color(0.05f, 0.06f, 0.08f, 0.85f));
        draw_rect(Rect2(legend_pos, legend_size), Color(0.3f, 0.4f, 0.5f, 0.5f), false, 1.0f);

        if (font.is_valid()) {
            String simd_info = _get_simd_architecture_name();
            int active_threads = WorkerThreadPool::get_singleton() ? WorkerThreadPool::get_singleton()->get_thread_count() : 1;

            String simd_text = vformat("SIMD : %s", simd_info);
            String thread_text = vformat("Threads : %d Workers", active_threads);

            draw_string(font, legend_pos + Vector2(10.0f, 18.0f), simd_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.3f, 0.85f, 1.0f));
            draw_string(font, legend_pos + Vector2(10.0f, 36.0f), thread_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.95f, 0.8f, 0.2f));

            draw_string(font, Vector2(margin, 16.0f), vformat("Execution Trend (Max: %.1fus)", max_time), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.9f, 0.9f, 0.9f));
        }
    }

public:
    void set_data(const Vector<double> &p_times) {
        iteration_times_us = p_times;
        queue_redraw();
    }
};

#endif // CAUSAL_GRAPH_CONTROL_H
