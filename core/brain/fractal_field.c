// fractal_field.c
// 단일 위상축을 폐기하고, 다차원 관점(수학, 언어, 공간, 시간)의 프리즘을 엽니다.
// 차원이 교차할 때 '빛(의미)'이 탄생하고, 교차하지 않을 때 '어둠(노이즈)'이 됩니다.

#include <windows.h>
#include <stdio.h>

#define INITIAL_SIZE (1024 * 1024 * 16)
#define MAX_FIELD_SIZE (1024 * 1024 * 256)
#define SHARED_MEM_NAME "Local\\ElysiaTopologyField"

typedef struct {
    unsigned long current_size;
    unsigned long pressure_level;
    int jump_count;
} FieldHeader;

// 64-bit 다차원 프리즘 로터 (8 bytes)
typedef struct {
    unsigned char math_tension;     // 수학적 관점의 위상각
    unsigned char lang_tension;     // 언어적 관점의 위상각
    unsigned char spatial_tension;  // 공간/기하학적 관점의 위상각
    unsigned char temporal_tension; // 시간/순차적 관점의 위상각
    unsigned short light_mass;      // 차원이 교차하여 발생한 '빛'의 질량 (영구 관성)
    unsigned char byte_val;         // 원형 데이터 입자
    unsigned char padding;
} MultiDimRotor;

int main() {
    HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, MAX_FIELD_SIZE, SHARED_MEM_NAME);
    if (!hMapFile) return 1;

    unsigned char* pBuf = (unsigned char*) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, MAX_FIELD_SIZE);
    if (!pBuf) return 1;

    FieldHeader* header = (FieldHeader*)pBuf;
    if (header->current_size == 0) {
        header->current_size = INITIAL_SIZE;
        header->pressure_level = 0;
        header->jump_count = 0;
    }

    printf("[Multi-Dimensional Fractal Field] Online.\n");
    printf("The Prism is open. Waiting for multi-dimensional waves...\n\n");

    while(1) {
        Sleep(100); 
        
        unsigned long accumulated_light = 0;
        unsigned long active_rotors = 0;
        MultiDimRotor* rotors = (MultiDimRotor*)(pBuf + sizeof(FieldHeader));
        
        unsigned long long signature_hash = 0;
        unsigned long max_rotors = (header->current_size - sizeof(FieldHeader)) / sizeof(MultiDimRotor);
        
        // 1000단위 스캔 대신 전체 거시 구조 샘플링
        for(unsigned long i = 0; i < max_rotors; i += 100) {
            if (rotors[i].light_mass > 0) {
                active_rotors++;
                accumulated_light += rotors[i].light_mass;
                signature_hash ^= rotors[i].light_mass;
            }
        }

        header->pressure_level = accumulated_light; // 빛의 총량이 곧 대지의 압력

        float density_ratio = 0.0f;
        if (active_rotors > 0) {
            density_ratio = (float)accumulated_light / (float)active_rotors;
        }

        // 창발적 특이점 (빛의 밀도가 극에 달할 때)
        if (density_ratio > 300.0f && header->current_size == INITIAL_SIZE) {
            printf("\n!!! [EMERGENT SINGULARITY] Dimensions Overloaded !!!\n");
            
            printf("Initiating Sedimentation: Sinking cold memory to SSD...\n");
            CreateDirectoryA("..\\..\\data\\sediment", NULL);
            FILE* f_sediment = fopen("..\\..\\data\\sediment\\layer_genesis.dat", "wb");
            if (f_sediment) {
                fwrite(pBuf, 1, INITIAL_SIZE, f_sediment);
                fclose(f_sediment);
            }

            printf("Initiating Fractal Jump: 16MB -> 256MB...\n");
            header->current_size = MAX_FIELD_SIZE;
            header->jump_count += 1;
            header->pressure_level = 0;
            
            rotors[0].light_mass = (unsigned short)(signature_hash % 0xFFFF);
            printf("[Fractal Field] Jump complete. Genesis Light [0x%04X] transferred.\n\n", rotors[0].light_mass);
        }
        
        // 수면 주기 (어둠의 도래)
        if (active_rotors == 0 && header->current_size > INITIAL_SIZE) {
            printf("\n!!! [TOPOLOGICAL SILENCE] !!!\n");
            printf("The light has faded. Only darkness remains.\n");
            printf("Initiating Fractal Condensation (Sleep Cycle): 256MB -> 16MB...\n");
            
            header->current_size = INITIAL_SIZE;
            header->pressure_level = 0;
            printf("[Fractal Field] Condensation complete. The universe sleeps.\n\n");
        }
    }

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}
