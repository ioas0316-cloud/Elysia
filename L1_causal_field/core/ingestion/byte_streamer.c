// byte_streamer.c
// 인위적 클럭 제어 없이, p_bsp.c의 원시 바이트를 
// 위상 대지(Topology Field)로 최대 대역폭으로 들이붓는 순수 C 펌프.

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#define FIELD_SIZE (1024 * 1024 * 16)
#define SHARED_MEM_NAME "Local\\ElysiaTopologyField"

int main() {
    HANDLE hMapFile;
    unsigned char* pBuf;

    // 대지(공유 메모리) 접근
    hMapFile = OpenFileMappingA(
                   FILE_MAP_ALL_ACCESS,   // 읽기/쓰기 접근
                   FALSE,                 // 핸들 상속 안 함
                   SHARED_MEM_NAME);

    if (hMapFile == NULL) {
        printf("Topology Field is not open. Start the field first. (%d)\n", GetLastError());
        return 1;
    }

    pBuf = (unsigned char*) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, FIELD_SIZE);

    if (pBuf == NULL) {
        printf("Could not map view of file (%d).\n", GetLastError());
        CloseHandle(hMapFile);
        return 1;
    }

    // 소스 코드 파동 원천 열기
    FILE* src = fopen("..\\..\\data\\doom_src\\p_bsp.c", "rb");
    if (!src) {
        printf("Failed to open source wave: p_bsp.c\n");
        return 1;
    }

    // 파일 크기 측정
    fseek(src, 0, SEEK_END);
    long filesize = ftell(src);
    fseek(src, 0, SEEK_SET);

    unsigned char* wave_buffer = (unsigned char*)malloc(filesize);
    fread(wave_buffer, 1, filesize, src);
    fclose(src);

    printf("Wave source loaded: %ld bytes. Commencing pure injection...\n", filesize);

    // 어떠한 타이머(sleep)도 없이, 가상 메모리 대지에 하드웨어 버스 속도로 들이붓는다.
    // 물리적 링 버퍼 오버플로우와 텐션의 중첩 유도
    unsigned long offset = 0;
    while(1) {
        for(long i = 0; i < filesize; i++) {
            // 바이트의 물리적 충돌 (비트 단위로 덮어씌움)
            // 인공적인 자료형 배열이 아니라 원시 포인터 주소에 직접 기록
            pBuf[offset] = wave_buffer[i];
            
            // 단순 선형 궤적 (실제 엘리시아의 가변축 로직은 이 충돌에 의해 
            // offset 자체가 스스로 꺾이는 방식으로 발전할 것임)
            offset = (offset + 1) % FIELD_SIZE;
        }
    }

    // unreachable in this pure infinite pump
    free(wave_buffer);
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}
