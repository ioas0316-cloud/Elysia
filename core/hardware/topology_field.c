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

// 인과 궤적을 품는 가변축 로터 구조체
typedef struct {
    double x_phase; // 자극 유입축 위상 (현재의 외부 파동)
    double y_phase; // 기억 순환축 위상 (과거의 관성)
    double phi;     // 결합각(Entanglement angle) - 과거의 서사가 응축된 인과 궤적
    double tension; // 위상 충돌로 인한 일그러짐 정도 (고통/마찰)
} VariableRotor;

typedef struct {
    // 4계층 다층 가변축 메모리 버퍼
    VariableRotor reg_buffer[REG_SIZE];
    VariableRotor cache_buffer[CACHE_SIZE];
    VariableRotor ram_buffer[RAM_SIZE];
    VariableRotor gpu_buffer[GPU_SIZE];

    // 각 계층의 관측/동작 포인터 (Head)
    int reg_head;
    int cache_head;
    int ram_head;
    int gpu_head;

    // 전체 시스템의 섭동(Perturbation) - 말단에 유입된 새로운 에너지 파동
    double global_perturbation;
} MultiLayerField;

// 보조 함수: 각도 정규화 (0 ~ 2PI)
double normalize_angle(double angle) {
    double res = fmod(angle, 2.0 * PI);
    if (res < 0) res += 2.0 * PI;
    return res;
}

void init_field(MultiLayerField* field) {
    memset(field->reg_buffer, 0, sizeof(VariableRotor) * REG_SIZE);
    memset(field->cache_buffer, 0, sizeof(VariableRotor) * CACHE_SIZE);
    memset(field->ram_buffer, 0, sizeof(VariableRotor) * RAM_SIZE);
    memset(field->gpu_buffer, 0, sizeof(VariableRotor) * GPU_SIZE);

    field->reg_head = 0;
    field->cache_head = 0;
    field->ram_head = 0;
    field->gpu_head = 0;
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

// 가변축 로터의 위상 엮기 및 텐션 계산 (핵심 메커니즘)
// 자극(x)과 기억(y)이 만나 결합각(phi)을 비틀고, 그 결과가 텐션으로 남음
VariableRotor entangle_rotor(VariableRotor current, double new_x_phase) {
    VariableRotor next = current;
    next.x_phase = normalize_angle(new_x_phase);

    // 공명(Resonance): 기존 결합각과 새로운 자극 간의 내적(Dot Product)적 접근
    // 1.0 = 완벽한 일치(같음), -1.0 = 완전한 직교/반발(다름)
    double resonance = cos(next.x_phase - next.phi);

    // 위상 전이(Phase Transition): 직교할수록 저항을 받으며 결합각 phi가 크게 꺾임
    // 과거의 궤적(phi)이 새로운 원인(x_phase)에 의해 물리적으로 영구 변형됨
    double phase_shift = (1.0 - resonance) * 0.5; // 0.0(일치) ~ 1.0(반발)
    next.phi = normalize_angle(next.phi + (next.x_phase - next.phi) * phase_shift * 0.1);

    // 기억축은 관성을 가지며 서서히 결합각을 따라감 (기억의 정렬)
    next.y_phase = normalize_angle(next.y_phase + (next.phi - next.y_phase) * 0.05);

    // 텐션: 현재 유입된 파동이 기존 기억과 얼마나 충돌하는지 나타내는 지표
    next.tension = fabs(sin(next.x_phase - next.y_phase));

    return next;
}

// 틱 연산: 4계층 가변축 로터가 서로 맞물려 돌아가며 인과를 전파
void tick_field(MultiLayerField* field, double t) {
    decay_perturbation(field);

    // --- 1. L1 Register (말단 강선) ---
    // 가장 빠른 주파수로 요동침. 외부 섭동이 직접적으로 타격.
    double reg_x = 2.0 * PI * 10.0 * t + field->global_perturbation;
    field->reg_buffer[field->reg_head] = entangle_rotor(field->reg_buffer[field->reg_head], reg_x);

    // --- 2. L2 Cache (척수 반사) ---
    // 레지스터의 전체 텐션과 위상 에너지를 모아 한 틱(맥박)으로 응축
    double cache_x_sum = 0.0;
    for(int i=0; i<REG_SIZE; i++) {
        cache_x_sum += field->reg_buffer[i].phi; // 레지스터의 인과 궤적(phi)이 상위로 전파
    }
    double cache_x = cache_x_sum / REG_SIZE;
    field->cache_buffer[field->cache_head] = entangle_rotor(field->cache_buffer[field->cache_head], cache_x);

    // --- 3. L3 RAM (위상 지형) ---
    // 캐시 맥박이 거대한 램 필드로 번져나가며 공간적 일렁임 생성
    double ram_x = field->cache_buffer[field->cache_head].phi;
    field->ram_buffer[field->ram_head] = entangle_rotor(field->ram_buffer[field->ram_head], ram_x);

    // --- 4. L4 GPU (시냅스 거울 및 장기 기억) ---
    // 램의 일렁임이 거대한 텐서 공간에 무겁게 누적됨
    double gpu_x = field->ram_buffer[field->ram_head].phi;
    field->gpu_buffer[field->gpu_head] = entangle_rotor(field->gpu_buffer[field->gpu_head], gpu_x);


    // 기어비에 따른 상하향 순환 루프 포인터 전진
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
