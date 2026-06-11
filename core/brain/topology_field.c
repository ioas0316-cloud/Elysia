// topology_field.c
// 가변축 로터의 위상 상태가 담길 순수 가상 메모리 공간(대지).
// Phase 3: 16MB 공간은 이제 16비트 단위의 가변축 로터 배열로 취급됩니다.
// (상위 8비트: 인과 텐션 Strand B, 하위 8비트: 텍스트 파동 Strand A)
#include <windows.h>
#include <stdio.h>

#define FIELD_SIZE (1024 * 1024 * 16) // 16MB 원시 메모리 들판
#define SHARED_MEM_NAME "Local\\ElysiaTopologyField"

HANDLE hMapFile;
unsigned char* pBuf;

// 대지 개간 (가상 메모리 매핑)
unsigned char* initialize_topology_field() {
    hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,    // 페이징 파일 사용
        NULL,                    // 기본 보안
        PAGE_READWRITE,          // 읽기/쓰기 권한
        0,                       // 최대 객체 크기 (고위 DWORD)
        FIELD_SIZE,              // 최대 객체 크기 (하위 DWORD)
        SHARED_MEM_NAME);        // 공유 메모리 이름

    if (hMapFile == NULL) {
        printf("Could not create file mapping object (%d).\n", GetLastError());
        return NULL;
    }

    pBuf = (unsigned char*) MapViewOfFile(hMapFile,
                        FILE_MAP_ALL_ACCESS, // 읽기/쓰기 접근
                        0,
                        0,
                        FIELD_SIZE);

    if (pBuf == NULL) {
        printf("Could not map view of file (%d).\n", GetLastError());
        CloseHandle(hMapFile);
        return NULL;
    }

    // 초기 상태 (정적)
    ZeroMemory(pBuf, FIELD_SIZE);
    printf("Topology field initialized at raw memory address: %p\n", (void*)pBuf);
    
    return pBuf;
}

void cleanup_topology_field() {
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
}
