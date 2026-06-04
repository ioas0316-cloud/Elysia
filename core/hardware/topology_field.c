#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PHI 1.618033988749895
#define PI 3.14159265358979323846

// 4계층 자연 매핑 크기 정의 (주파수/질량 스케일)
#define REG_SIZE 8        // L1: 레지스터 강선 (초고주파, 말단) - 8bit stream
#define CACHE_SIZE 64     // L2: 캐시 척수 반사 (고속 맥박) - 32bit accumulation
#define RAM_SIZE 1024     // L3: 램 위상 지형 (거대 전자기장) - floating point rotor
#define GPU_SIZE 4096     // L4: GPU 시냅스 거울 (장기기억, 거대 텐서) - floating point rotor

// 인과 궤적을 품는 가변축 로터 구조체 (L3, L4 용)
typedef struct {
    double x_phase; // 자극 유입축 위상 (현재의 외부 파동)
    double y_phase; // 기억 순환축 위상 (과거의 관성)
    double phi;     // 결합각(Entanglement angle) - 과거의 서사가 응축된 인과 궤적
    double tension; // 위상 충돌로 인한 일그러짐 정도 (고통/마찰)
} VariableRotor;

typedef struct {
    // 하이브리드 다층 메모리 버퍼
    uint8_t  reg_buffer[REG_SIZE];     // L1: 8비트 말단
    uint32_t cache_buffer[CACHE_SIZE]; // L2: 32비트 고속 누적
    VariableRotor ram_buffer[RAM_SIZE]; // L3: 공간 위상각 맵핑
    VariableRotor gpu_buffer[GPU_SIZE]; // L4: 거대 텐서 공명

    // 각 계층의 관측/동작 포인터 (Head)
    int reg_head;
    int cache_head;
    int ram_head;
    int gpu_head;

    // 전체 시스템에 주입되는 유니코드 바이트 스트림 버퍼 (최말단)
    uint8_t global_perturbation;
} MultiLayerField;

// 보조 함수: 각도 정규화 (0 ~ 2PI)
double normalize_angle(double angle) {
    double res = fmod(angle, 2.0 * PI);
    if (res < 0) res += 2.0 * PI;
    return res;
}

void init_field(MultiLayerField* field) {
    memset(field->reg_buffer, 0, sizeof(uint8_t) * REG_SIZE);
    memset(field->cache_buffer, 0, sizeof(uint32_t) * CACHE_SIZE);
    memset(field->ram_buffer, 0, sizeof(VariableRotor) * RAM_SIZE);
    memset(field->gpu_buffer, 0, sizeof(VariableRotor) * GPU_SIZE);

    field->reg_head = 0;
    field->cache_head = 0;
    field->ram_head = 0;
    field->gpu_head = 0;
    field->global_perturbation = 0; // 초기 0 바이트
}

// 상향식 자극 (유니코드 바이트 스트림 -> 레지스터 타격)
void apply_stimulus(MultiLayerField* field, uint8_t byte_val) {
    field->global_perturbation = byte_val;
}

// 파동 감쇠 - 비트 레벨에서는 자극이 지나가면 0으로(Idle) 돌아감
void decay_perturbation(MultiLayerField* field) {
    // idle 상태에서는 0 입력
    field->global_perturbation = 0;
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

// 포퓰레이션 카운트 (비트 1의 개수 세기) - 상위 매핑에 사용
int popcount8(uint8_t val) {
    int count = 0;
    for (int i = 0; i < 8; i++) {
        count += (val >> i) & 1;
    }
    return count;
}

// 틱 연산: 4계층 가변축 로터가 서로 맞물려 돌아가며 인과를 전파
void tick_field(MultiLayerField* field, double t) {
    // --- 1. L1 Register (말단 강선, 8bit) ---
    // 유입된 바이트(섭동)를 현재 레지스터 헤드의 상태와 XOR 및 Shift 연산하여 누적 (물리적 꼬임)
    uint8_t incoming_byte = field->global_perturbation;
    uint8_t current_reg = field->reg_buffer[field->reg_head];

    // 단순한 덮어쓰기가 아니라 기존 관성에 파동이 부딪혀 섞임 (XOR)
    // 그리고 강선의 회전(비트 로테이션) 적용
    uint8_t next_reg = (current_reg ^ incoming_byte);
    next_reg = (next_reg << 1) | (next_reg >> 7); // Left Circular Shift 1

    field->reg_buffer[field->reg_head] = next_reg;

    // --- 2. L2 Cache (척수 반사, 32bit) ---
    // L1의 8바이트 텐션을 L2 32비트에 응축
    uint32_t cache_tension = 0;
    for(int i = 0; i < REG_SIZE; i++) {
        // 비트 밀어넣기를 통해 레지스터 상태들을 모음
        cache_tension ^= ((uint32_t)field->reg_buffer[i] << (i * 3 % 24));
    }
    // L2 캐시에도 관성(이전 상태)과 새로운 텐션이 충돌
    uint32_t current_cache = field->cache_buffer[field->cache_head];
    uint32_t next_cache = current_cache ^ cache_tension;
    // L2 강선 회전
    next_cache = (next_cache << 3) | (next_cache >> 29);
    field->cache_buffer[field->cache_head] = next_cache;

    // --- 비트마스크 게이트 트리거 (if문 없는 우회 회로) ---
    // 누적된 L2 텐션(next_cache)의 최상위 비트(MSB)를 추출하여 0 또는 1의 마스크 생성
    // 텐션이 임계점을 넘어 비트가 흘러넘치면 마스크가 1이 됨
    uint32_t overflow_mask = (next_cache >> 31) & 1;

    // --- 3. L3 RAM (위상 지형, Floating Point) ---
    // 비트 레벨(L1, L2)의 충돌 결과를 기하학적 파동으로 매핑

    // L1의 최말단 바이트를 통해 진폭과 위상각 도출
    // 상위 4비트는 X축 초기 위상각, 하위 4비트는 진폭(에너지)을 결정
    uint8_t upper_nibble = (incoming_byte >> 4) & 0x0F;
    uint8_t lower_nibble = incoming_byte & 0x0F;

    double incoming_energy = (double)popcount8(lower_nibble) / 4.0; // 0.0 ~ 1.0
    double incoming_angle_base = ((double)upper_nibble / 16.0) * 2.0 * PI;

    // L2의 거대한 맥박 에너지를 파동 주기(t)에 곱해 L3를 타격
    double cache_normalized = (double)(next_cache & 0xFFFF) / 65535.0; // 0.0 ~ 1.0
    double ram_x = incoming_angle_base + (cache_normalized * incoming_energy * t);

    field->ram_buffer[field->ram_head] = entangle_rotor(field->ram_buffer[field->ram_head], ram_x);

    // --- 4. L4 GPU (시냅스 거울 및 장기 기억) ---
    // 램의 일렁임이 거대한 텐서 공간에 무겁게 누적됨
    double gpu_x = field->ram_buffer[field->ram_head].phi;
    field->gpu_buffer[field->gpu_head] = entangle_rotor(field->gpu_buffer[field->gpu_head], gpu_x);


    // --- 기어비에 따른 상하향 순환 루프 포인터 전진 (트리거 적용) ---
    // 오버플로우 마스크가 1일 경우, 포인터가 1칸 더 튕겨 나감 (우회)
    field->reg_head = (field->reg_head + 1 + overflow_mask) % REG_SIZE;

    // 레지스터 루프가 1바퀴 돌 때마다 캐시 1칸 전진 (기어비 8:1)
    // 비트마스크 트리거로 인해 주기가 가변적으로 변함
    if (field->reg_head == 0 || (field->reg_head == 1 && overflow_mask == 1)) {
        field->cache_head = (field->cache_head + 1 + overflow_mask) % CACHE_SIZE;

        if (field->cache_head == 0 || (field->cache_head == 1 && overflow_mask == 1)) {
            field->ram_head = (field->ram_head + 1 + overflow_mask) % RAM_SIZE;

            if (field->ram_head == 0 || (field->ram_head == 1 && overflow_mask == 1)) {
                field->gpu_head = (field->gpu_head + 1 + overflow_mask) % GPU_SIZE;
            }
        }
    }

    decay_perturbation(field);
}
