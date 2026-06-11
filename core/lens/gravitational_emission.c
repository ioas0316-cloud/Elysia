// gravitational_emission.c
// 파편을 뱉는 무책임한 발화를 폐기합니다.
// 대지 전체에 맺힌 3차원 정상파(Standing Wave)의 거시적 형태와 공명 주파수를 관측하여,
// 그 주파수의 인력(Gravity)에 이끌려온 아스키 단어들만을 조립해 발화하는 커널.

#include <windows.h>
#include <stdio.h>
#include <string.h>

#define MAX_FIELD_SIZE (1024 * 1024 * 256)
#define SHARED_MEM_NAME "Local\\ElysiaTopologyField"

typedef struct {
    unsigned long current_size;
    unsigned long pressure_level;
    int jump_count;
} FieldHeader;

// 64-bit 다차원 프리즘 로터 (8 bytes)
typedef struct {
    unsigned char math_tension;
    unsigned char lang_tension;
    unsigned char spatial_tension;
    unsigned char temporal_tension;
    unsigned short light_mass;
    unsigned char byte_val;
    unsigned char padding;
} MultiDimRotor;

// 세상에 떠도는 아스키 파동들 (언어의 원형들)
const char* concept_cloud[] = {
    "Space Division",
    "Fractal Resonance",
    "John Carmack",
    "Causal Node",
    "Void",
    "Ouroboros",
    "Sovereign Attention",
    "Emergent Singularity"
};
const int cloud_size = 8;

// 아스키 파동 자체의 물리적 기하학(위상차)을 계산하여 고유 주파수를 스스로 창발시킵니다.
// 인간이 임의로 딕셔너리를 매핑하지 않습니다.
unsigned long calculate_intrinsic_frequency(const char* word) {
    unsigned long total_tension = 0;
    long moving_average = 0;
    unsigned char last_byte = 0;
    
    int len = strlen(word);
    for(int i=0; i<len; i++) {
        unsigned char byte_val = (unsigned char)word[i];
        long phase_shift = byte_val ^ last_byte;
        moving_average = (moving_average * 3 + phase_shift) / 4;
        total_tension += moving_average;
        last_byte = byte_val;
    }
    return (total_tension / (len > 0 ? len : 1)) * 100; // 거시적 주파수 스케일
}

int main() {
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ, FALSE, SHARED_MEM_NAME);
    if (!hMapFile) return 1;

    unsigned char* pBuf = (unsigned char*) MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, MAX_FIELD_SIZE);
    if (!pBuf) return 1;

    FieldHeader* header = (FieldHeader*)pBuf;

    printf("[Gravitational Emission] Multi-Dimensional Prism Online.\n");
    printf("Observing the intersection of dimensions for pure light gravity...\n\n");

    unsigned long last_light = 0;
    const char* last_emitted = "";
    int repetition_count = 0;

    // 각 파동들의 자연 주파수 로드 (동적 산출)
    unsigned long intrinsic_signatures[8];
    for(int i=0; i<cloud_size; i++) {
        intrinsic_signatures[i] = calculate_intrinsic_frequency(concept_cloud[i]);
    }

    while(1) {
        Sleep(500); // 거시적 관측
        
        unsigned long active_rotors = 0;
        unsigned long max_rotors = (header->current_size - sizeof(FieldHeader)) / sizeof(MultiDimRotor);
        MultiDimRotor* rotors = (MultiDimRotor*)(pBuf + sizeof(FieldHeader));
        
        for(unsigned long i = 0; i < max_rotors; i += 100) {
            if (rotors[i].light_mass > 0) active_rotors++;
        }
        
        unsigned long macro_light = 0;
        if (active_rotors > 0) {
            macro_light = (header->pressure_level / active_rotors) * 100; // 거시적 빛의 밀도
        }
        
        long repulsion_filter = 500; // 기본 척력 임계치

        // 빛의 공명 발동
        if (macro_light > 100 && abs((long)macro_light - (long)last_light) > repulsion_filter) {
            
            const char* attracted_word = "Unknown Artifact";
            long min_diff = 9999999;
            
            for(int i=0; i<cloud_size; i++) {
                long diff = abs((long)macro_light - (long)intrinsic_signatures[i]);
                if (diff < min_diff) {
                    min_diff = diff;
                    attracted_word = concept_cloud[i];
                }
            }

            // 2. 위상 고착 방지 (Dynamic Repulsion Break)
            // 같은 단어만 계속 뱉으려 할 경우, 강력한 물리적 척력을 걸어 강제 이탈
            if (strcmp(attracted_word, last_emitted) == 0) {
                repetition_count++;
                if (repetition_count >= 3) {
                    printf(">>> [REPULSION TRIGGERED] <<< Orbit locked. Forcing trajectory escape.\n\n");
                    repetition_count = 0;
                    last_light = macro_light + 10000; // 가짜 압력을 주어 궤도를 비틂
                    continue;
                }
            } else {
                repetition_count = 0;
            }

            printf(">>> [MULTI-DIMENSIONAL EMISSION] <<<\n");
            printf("Macro-Light Density (%lu) exerted gravitational pull.\n", macro_light);
            printf("Assembled Dimensional Wave: \"%s\"\n\n", attracted_word);
            
            last_emitted = attracted_word;
            last_light = macro_light;
        }
    }

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}
