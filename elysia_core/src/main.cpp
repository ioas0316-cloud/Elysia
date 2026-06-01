#include <iostream>
#include <iomanip>
#include "CoreKernel.hpp"
#include "EnvironmentGenerator.hpp"

int main() {
    std::cout << "========================================\n";
    std::cout << " Elysia Core PoC - The Digital Walk \n";
    std::cout << "========================================\n\n";

    CoreKernel kernel;
    EnvironmentGenerator env_gen;

    const int MAX_TICKS = 50;
    double t = 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Tick |   Time | Env Data |  Phase | Resist | Resonance(Feedback)\n";
    std::cout << "----------------------------------------------------------------\n";

    for (int i = 0; i < MAX_TICKS; ++i) {
        // 1. 현재 시간에 따른 환경(주파수 파형) 데이터 관측
        double env_data = env_gen.get_data(t);

        // 2. 코어 커널 1틱 구동 (걷기 수행 및 자기인식 피드백 루프)
        CoreKernel::Status status = kernel.tick(env_data);

        // 3. 상태 출력
        std::cout << std::setw(4) << i << " | "
                  << std::setw(6) << t << " | "
                  << std::setw(8) << status.environment_value << " | "
                  << std::setw(6) << status.current_phase << " | "
                  << std::setw(6) << status.knob_resistance << " | "
                  << std::setw(8) << status.cognitive_resonance << "\n";

        // 시간 전진
        t += 0.1;
    }

    std::cout << "\n========================================\n";
    std::cout << " Simulation Complete. The walker continues.\n";
    std::cout << "========================================\n";

    return 0;
}
