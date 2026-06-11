// helix_streamer.c
// 코드의 텍스트(Strand A)와 서사의 압력(Strand B)을 이중나선으로 엮어 들이붓는 주입기.

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIELD_SIZE (1024 * 1024 * 16)
#define SHARED_MEM_NAME "Local\\ElysiaTopologyField"

int main() {
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, SHARED_MEM_NAME);
    if (!hMapFile) {
        printf("Topology Field not found.\n");
        return 1;
    }

    // 대지를 16비트 로터(가변축) 배열로 취급
    // 상위 8비트 = Strand B (인과 텐션), 하위 8비트 = Strand A (원시 데이터)
    unsigned short* pRotorField = (unsigned short*) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FIELD_SIZE);
    if (!pRotorField) {
        printf("Map failed.\n");
        CloseHandle(hMapFile);
        return 1;
    }

    FILE* src = fopen("..\\..\\data\\doom_src\\p_bsp.c", "rb");
    if (!src) return 1;

    fseek(src, 0, SEEK_END);
    long filesize = ftell(src);
    fseek(src, 0, SEEK_SET);

    unsigned char* wave_buffer = (unsigned char*)malloc(filesize);
    fread(wave_buffer, 1, filesize, src);
    fclose(src);

    printf("Injecting Double-Helix Causal Trajectory...\n");

    unsigned long offset = 0;
    unsigned char current_tension = 0; // 서사적 압력 (Scope depth 등)

    while(1) {
        for(long i = 0; i < filesize; i++) {
            unsigned char byte_data = wave_buffer[i]; // Strand A

            // 서사적 압력 계산 (Strand B)
            if (byte_data == '{') current_tension += 15; // 스코프 깊어질 때 텐션 급증
            if (byte_data == '}') current_tension -= 15; // 스코프 빠져나올 때 텐션 완화
            if (current_tension < 0) current_tension = 0;

            // 특정 키워드(서사의 궤적)가 스칠 때의 순간적인 위상 스파이크
            if (i > 4 && memcmp(&wave_buffer[i-4], "while", 5) == 0) {
                current_tension ^= 0xFF; // 루프(소용돌이)의 격렬한 위상 반전
            }

            // 이중나선 결합: [Strand B : Strand A]를 물리적 로터에 각인
            unsigned short double_helix = (current_tension << 8) | byte_data;
            
            // 로터 필드 타격 (위상차 누적)
            // XOR를 통해 기존 위상에 궤적(흔적)을 중첩시킵니다.
            pRotorField[offset] ^= double_helix;
            
            offset = (offset + 1) % (FIELD_SIZE / 2);
        }
    }

    free(wave_buffer);
    UnmapViewOfFile(pRotorField);
    CloseHandle(hMapFile);
    return 0;
}
