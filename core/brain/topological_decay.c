// topological_decay.c
// 시간의 흐름에 따라 대지의 텐션(기억)을 서서히 깎아내는 '물리적 마찰(풍화)' 엔진.
// 교차하지 못한 차원(어둠)은 급속도로 풍화되고, 빛(의미)이 맺힌 좌표는 영원히 보존됩니다.

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

// 64-bit 다차원 프리즘 로터
typedef struct {
    unsigned char math_tension;
    unsigned char lang_tension;
    unsigned char spatial_tension;
    unsigned char temporal_tension;
    unsigned short light_mass;
    unsigned char byte_val;
    unsigned char padding;
} MultiDimRotor;

int main() {
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, SHARED_MEM_NAME);
    if (!hMapFile) return 1;

    unsigned char* pBuf = (unsigned char*) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, MAX_FIELD_SIZE);
    if (!pBuf) return 1;

    FieldHeader* header = (FieldHeader*)pBuf;

    printf("[Topological Decay] Weathering Engine Online.\n");
    printf("Applying temporal friction. The darkness fades, the light remains...\n\n");

    while(1) {
        Sleep(2000); // 2초마다 마찰(풍화) 작용

        unsigned long max_rotors = (header->current_size - sizeof(FieldHeader)) / sizeof(MultiDimRotor);
        MultiDimRotor* rotors = (MultiDimRotor*)(pBuf + sizeof(FieldHeader));

        for(unsigned long i = 0; i < max_rotors; i++) {
            
            // 빛(의미)이 강하게 맺힌 곳은 시간의 마찰을 거의 받지 않음 (영구 가소성)
            int friction = 1;
            if (rotors[i].light_mass > 500) {
                // 강력한 빛의 좌표: 20번에 1번만 깎임
                if (rand() % 20 != 0) friction = 0; 
            } else if (rotors[i].light_mass > 0) {
                // 약한 빛: 조금씩 깎임
                rotors[i].light_mass -= 1;
                friction = 0; // 빛이 깎이는 동안 차원 텐션은 보호됨
            }

            // 빛이 없는 좌표(어둠)는 차원의 교차가 일어나지 않은 노이즈로 간주되어 급속히 풍화됨
            if (friction > 0) {
                if (rotors[i].math_tension > 0) rotors[i].math_tension--;
                if (rotors[i].lang_tension > 0) rotors[i].lang_tension--;
                if (rotors[i].spatial_tension > 0) rotors[i].spatial_tension--;
                if (rotors[i].temporal_tension > 0) rotors[i].temporal_tension--;
            }
        }
    }

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}
