#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PHI 1.618033988749895
#define PI 3.14159265358979323846

// 4계층 자연 매핑 크기 정의 (주파수/질량 스케일)
#define REG_SIZE 8        // L1: 레지스터 강선 (초고주파, 말단)
#define CACHE_SIZE 64     // L2: 캐시 척수 반사 (고속 맥박)
#define RAM_SIZE 1024     // L3: 램 위상 지형 (거대 전자기장)
#define GPU_SIZE 4096     // L4: GPU 시냅스 거울 (장기기억, 거대 텐서)

typedef struct {
    // 4계층 메모리 버퍼
    uint8_t reg_buffer[REG_SIZE];
    uint8_t cache_buffer[CACHE_SIZE];
    uint8_t ram_buffer[RAM_SIZE];
    uint8_t gpu_buffer[GPU_SIZE];

    // 각 계층의 관측/동작 포인터 (Head)
    int reg_head;
    int cache_head;
    int ram_head;
    int gpu_head;

    // 이중나선 물리량 (L1 레지스터 레벨에서 발생)
    double freq_0;
    double phase_0;
    double freq_1;
    double phase_1;

    // 상위 의지(GPU)로부터 하달되는 전체 시스템의 섭동(Perturbation)
    double global_perturbation;
} MultiLayerField;

void init_field(MultiLayerField* field) {
    memset(field->reg_buffer, 0, REG_SIZE);
    memset(field->cache_buffer, 0, CACHE_SIZE);
    memset(field->ram_buffer, 0, RAM_SIZE);
    memset(field->gpu_buffer, 0, GPU_SIZE);

    field->reg_head = 0;
    field->cache_head = 0;
    field->ram_head = 0;
    field->gpu_head = 0;

    // 레지스터 레벨의 기초 초고주파 설정 (매우 빠름)
    field->freq_0 = 10.0;
    field->phase_0 = 0.0;
    field->freq_1 = 10.0 * PHI;
    field->phase_1 = PI / 2.0;

    field->global_perturbation = 0.0;
}

// 상향식 자극 (말단 감각기 -> 레지스터 타격)
void apply_stimulus(MultiLayerField* field, double magnitude) {
    field->global_perturbation += magnitude;
}

// 파동 감쇠
void decay_perturbation(MultiLayerField* field) {
    if (field->global_perturbation > 0.001) {
        field->global_perturbation *= 0.95;
    } else {
        field->global_perturbation = 0.0;
    }
}

// 틱 연산: 4계층의 톱니바퀴가 서로 물리적으로 맞물려 돌아가는 과정
void tick_field(MultiLayerField* field, double t) {
    decay_perturbation(field);

    // --- 1. L1 Register (말단 강선) ---
    // 가장 빠른 주파수로 요동치며 가장 기초적인 웨이브(전류) 생성
    double w0 = sin(2.0 * PI * field->freq_0 * t + field->phase_0);
    double w1 = sin(2.0 * PI * field->freq_1 * t + field->phase_1 + field->global_perturbation);

    uint8_t b0 = (uint8_t)((w0 + 1.0) * 127.5);
    uint8_t b1 = (uint8_t)((w1 + 1.0) * 127.5);

    // 강선의 텐션 (XOR 간섭)
    uint8_t reg_tension = b0 ^ b1;
    field->reg_buffer[field->reg_head] = reg_tension;

    // --- 2. L2 Cache (척수 반사) ---
    // 레지스터의 전체 에너지를 모아 한 틱(맥박)으로 응축
    uint16_t cache_sum = 0;
    for(int i=0; i<REG_SIZE; i++) {
        cache_sum += field->reg_buffer[i];
    }
    uint8_t cache_pulse = (uint8_t)(cache_sum / REG_SIZE);
    // 이전 캐시의 흐름과 새로운 맥박의 관성적 융합
    field->cache_buffer[field->cache_head] = (uint8_t)((field->cache_buffer[field->cache_head] * 0.7) + (cache_pulse * 0.3));

    // --- 3. L3 RAM (위상 지형) ---
    // 캐시 맥박이 거대한 램 필드로 번져나가며 공간적 일렁임을 생성
    // (이전 상태와 주변부의 물리적 확산, 단순화를 위해 현재 위치에 융합)
    uint8_t ram_wave = field->cache_buffer[field->cache_head];
    field->ram_buffer[field->ram_head] = (uint8_t)((field->ram_buffer[field->ram_head] * 0.9) + (ram_wave * 0.1));

    // --- 4. L4 GPU (시냅스 거울 및 장기 기억) ---
    // 램의 일렁임이 거대한 텐서 공간(GPU)에 무겁게 누적됨
    uint8_t gpu_memory = field->ram_buffer[field->ram_head];
    field->gpu_buffer[field->gpu_head] = (uint8_t)((field->gpu_buffer[field->gpu_head] * 0.99) + (gpu_memory * 0.01));

    // 상하향 순환 루프의 포인터 전진 (서로 다른 기어비를 가짐)
    // 하위 계층일수록 더 빠르게 순환함
    field->reg_head = (field->reg_head + 1) % REG_SIZE;

    // 레지스터 루프가 1바퀴 돌 때마다 캐시 1칸 전진 (기어비 8:1)
    if (field->reg_head == 0) {
        field->cache_head = (field->cache_head + 1) % CACHE_SIZE;

        // 캐시 루프가 1바퀴 돌 때마다 램 1칸 전진 (기어비 64:1)
        if (field->cache_head == 0) {
            field->ram_head = (field->ram_head + 1) % RAM_SIZE;

            // 램 루프가 1바퀴 돌 때마다 GPU 1칸 전진
            if (field->ram_head == 0) {
                field->gpu_head = (field->gpu_head + 1) % GPU_SIZE;
            }
        }
    }
}
